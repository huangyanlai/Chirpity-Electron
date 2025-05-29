let tf;

try {
  tf = require("@tensorflow/tfjs-node");
} catch {
  tf = require("@tensorflow/tfjs");
}

const DEBUG = false;
class BaseModel {
  constructor(appPath, version) {
    this.model = undefined;
    this.labels = undefined;
    this.height = undefined;
    this.width = undefined;
    this.config = { sampleRate: 24_000, specLength: 3, sigmoid: 1 };
    this.chunkLength = this.config.sampleRate * this.config.specLength;
    this.overlap = 0;
    this.model_loaded = false;
    this.frame_length = 512;
    this.frame_step = 186;
    this.appPath = appPath;
    this.useContext = undefined;
    this.version = version;
    this.selection = false;
    this.scalarFive = tf.scalar(5);
  }

  async loadModel(type) {
    DEBUG && console.log("loading model");
    if (this.model_loaded === false) {
      // Model files must be in a different folder than the js, assets files
      DEBUG && console.log("loading model from", this.appPath + "model.json");
      const load = type === "layers" ? tf.loadLayersModel : tf.loadGraphModel;
      this.model = await load(this.appPath + "model.json", {
        weightPathPrefix: this.appPath,
      });
      this.model_loaded = true;
      this.inputShape = [...this.model.inputs[0].shape];
    }
  }

  async warmUp(batchSize) {
    this.batchSize = parseInt(batchSize);
    this.inputShape[0] = this.batchSize;
    DEBUG && console.log("WarmUp begin", tf.memory().numTensors);
    const input = tf.zeros(this.inputShape);

    // Parallel compilation for faster warmup
    // https://github.com/tensorflow/tfjs/pull/7755/files#diff-a70aa640d286e39c922aa79fc636e610cae6e3a50dd75b3960d0acbe543c3a49R316
    if (tf.getBackend() === "webgl") {
      tf.env().set("ENGINE_COMPILE_ONLY", true);
      const compileRes = this.model.predict(input);
      tf.env().set("ENGINE_COMPILE_ONLY", false);
      await tf.backend().checkCompileCompletionAsync();
      tf.backend().getUniformLocations();
      tf.dispose(compileRes);
      input.dispose();
    } else if (tf.getBackend() === "webgpu") {
      tf.env().set("WEBGPU_ENGINE_COMPILE_ONLY", true);
      const compileRes = this.model.predict(input);
      await tf.backend().checkCompileCompletionAsync();
      tf.dispose(compileRes);
      tf.env().set("WEBGPU_ENGINE_COMPILE_ONLY", false);
    } else {
      // Tensorflow backend
      // const compileRes = this.model.predict(input);
      // tf.dispose(compileRes);
    }
    input.dispose();
    DEBUG && console.log("WarmUp end", tf.memory().numTensors);
    return true;
  }

  normalise = (spec) => spec.mul(255).div(spec.max([1, 2], true));

  padBatch(tensor) {
    return tf.tidy(() => {
      DEBUG &&
        console.log(
          `Adding ${this.batchSize - tensor.shape[0]} tensors to the batch`
        );
      const shape = [...tensor.shape];
      shape[0] = this.batchSize - shape[0];
      const padding = tf.zeros(shape);
      return tf.concat([tensor, padding], 0);
    });
  }

  addAudioContext(prediction, tensor, confidence) {
    return tf.tidy(() => {
      // Create a set of audio segments from the batch, offset by half the length of the original audio
      const [batchSize, length] = tensor.shape;
      const halfLength = Math.floor(length / 2);
      const firstHalf = tensor.slice([0, 0], [-1, halfLength]);
      const secondHalf = tensor.slice([0, halfLength], [-1, halfLength]);
      const paddedSecondHalf = tf.concat(
        [tf.zeros([batchSize, halfLength]), secondHalf],
        1
      );
      secondHalf.dispose();
      // prepend padding tensor
      const [droppedSecondHalf, _] = paddedSecondHalf.split([
        paddedSecondHalf.shape[0],
        1,
      ]); // pop last tensor
      paddedSecondHalf.dispose();
      const combined = tf.concat([droppedSecondHalf, firstHalf], 1); // concatenate adjacent pairs along the time dimension
      firstHalf.dispose();
      droppedSecondHalf.dispose();
      const rshiftPrediction = this.model.predict(combined, {
        batchSize: this.batchSize,
      });
      combined.dispose();
      // now we have predictions for both the original and rolled audio segments
      const [padding, remainder] = tf.split(rshiftPrediction, [1, -1]);
      const lshiftPrediction = tf.concat([remainder, padding]);
      // Get the highest predictions from the overlapping segments
      const surround = tf.maximum(rshiftPrediction, lshiftPrediction);
      lshiftPrediction.dispose();
      rshiftPrediction.dispose();
      // Mask out where these are below the threshold
      const indices = tf.greater(surround, confidence);
      return prediction.where(indices, 0);
    });
  }
  addContext(prediction, tensor, confidence) {
    if (tensor.shape.length < 4) {
      return this.addAudioContext(prediction, tensor, confidence);
    }
    // Create a set of images from the batch, offset by half the width of the original images
    const [_, height, width, channel] = tensor.shape;
    return tf.tidy(() => {
      const firstHalf = tensor.slice([0, 0, 0, 0], [-1, -1, width / 2, -1]);
      const secondHalf = tensor.slice(
        [0, 0, width / 2, 0],
        [-1, -1, width / 2, -1]
      );
      const paddedSecondHalf = tf.concat(
        [tf.zeros([1, height, width / 2, channel]), secondHalf],
        0
      );
      secondHalf.dispose();
      // prepend padding tensor
      const [droppedSecondHalf, _] = paddedSecondHalf.split([
        paddedSecondHalf.shape[0] - 1,
        1,
      ]); // pop last tensor
      paddedSecondHalf.dispose();
      const combined = tf.concat([droppedSecondHalf, firstHalf], 2); // concatenate adjacent pairs along the width dimension
      firstHalf.dispose();
      droppedSecondHalf.dispose();
      const rshiftPrediction = this.model.predict(combined, {
        batchSize: this.batchSize,
      });
      combined.dispose();
      // now we have predictions for both the original and rolled images
      const [padding, remainder] = tf.split(rshiftPrediction, [1, -1]);
      const lshiftPrediction = tf.concat([remainder, padding]);
      // Get the highest predictions from the overlapping images
      const surround = tf.maximum(rshiftPrediction, lshiftPrediction);
      lshiftPrediction.dispose();
      rshiftPrediction.dispose();
      // Mask out where these are below the threshold
      const indices = tf.greater(surround, confidence);
      return prediction.where(indices, 0);
    });
  }
  async predictBatch(audio, keys, threshold, confidence) {
    const prediction = this.model.predict(audio, { batchSize: this.batchSize });
    let newPrediction;
    if (this.selection) {
      newPrediction = tf.max(prediction, 0, true);
      prediction.dispose();
      keys = keys.splice(0, 1);
    } else if (this.useContext && this.batchSize > 1 ) {
      newPrediction = this.addContext(prediction, audio, confidence);
      prediction.dispose();
    }

    audio.dispose();
    const finalPrediction = newPrediction || prediction;

    const { indices, values } = tf.topk(finalPrediction, 5, true);
    finalPrediction.dispose();
    // The GPU backend is *so* slow with BirdNET, let's not queue up predictions
    const [topIndices, topValues] = await Promise.all([
      indices.array(),
      values.array(),
    ]).catch((err) => console.log("Data transfer error:", err));
    indices.dispose();
    values.dispose();

    keys = keys.map((key) => (key / this.config.sampleRate).toFixed(3));
    return [keys, topIndices, topValues];
  }

  makeSpectrogram(signal) {
    return tf.tidy(() => {
      let spec = tf.abs(
        tf.signal.stft(signal, this.frame_length, this.frame_step)
      );
      signal.dispose();
      return spec;
    });
  }

  fixUpSpecBatch(specBatch, h, w) {
    const img_height = h || this.height;
    const img_width = w || this.width;
    return tf.tidy(() => {
      // Preprocess tensor

      specBatch = specBatch
        .slice([0, 0, 0], [-1, img_width, img_height])
        .transpose([0, 2, 1])
        .reverse([1]);

      // Split into main part and bottom rows
      const [mainPart, bottomRows] = tf.split(
        specBatch,
        [img_height - 10, 10],
        1
      );

      // Concatenate after adjusting bottom rows
      return this.normalise(
        tf.concat([mainPart, bottomRows.div(this.scalarFive)], 1)
      ).expandDims(-1);
    });
  }

  padAudio = (audio) => {
    const step = this.chunkLength - (this.overlap * this.config.sampleRate);
    const totalChunks = Math.ceil((audio.length - this.chunkLength) / step) + 1;
    const requiredLength = (totalChunks - 1) * step + this.chunkLength;
  
    if (audio.length < requiredLength) {
      const paddedAudio = new Float32Array(requiredLength);
      paddedAudio.set(audio);
      return paddedAudio;
    } else {
      return audio;
    }
  };

  createAudioTensorBatch = (audio) => {
    return tf.tidy(() => {
      audio = this.padAudio(audio); // Make sure it's padded appropriately
      const step = this.chunkLength - (this.overlap * this.config.sampleRate);
      const totalLength = audio.length;
  
      // Convert to tensor
      const audioTensor = tf.tensor1d(audio);
  
      const chunks = [];
      for (let start = 0; start + this.chunkLength <= totalLength; start += step) {
        const chunk = audioTensor.slice(start, this.chunkLength);
        chunks.push(chunk);
      }
  
      const batch = tf.stack(chunks);
      return [batch, chunks.length];
    });
  };

  getKeys = (numSamples, start) =>
    [...Array(numSamples).keys()].map((i) => start + (this.chunkLength - (this.overlap * this.config.sampleRate)) * i);
}
export { BaseModel };
