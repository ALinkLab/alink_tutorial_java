package com.alibaba.alink;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.io.filesystem.FilePath;
import com.alibaba.alink.common.io.filesystem.FileSystemUtils;
import com.alibaba.alink.common.io.filesystem.LocalFileSystem;
import com.alibaba.alink.common.utils.Stopwatch;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.audio.ExtractMfccFeatureBatchOp;
import com.alibaba.alink.operator.batch.audio.ReadAudioToTensorBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalMultiClassBatchOp;
import com.alibaba.alink.operator.batch.sink.AkSinkBatchOp;
import com.alibaba.alink.operator.batch.source.AkSourceBatchOp;
import com.alibaba.alink.operator.batch.source.MemSourceBatchOp;
import com.alibaba.alink.params.dataproc.TensorToVectorParams.ConvertMethod;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.audio.ExtractMfccFeature;
import com.alibaba.alink.pipeline.classification.KerasSequentialClassifier;
import com.alibaba.alink.pipeline.classification.Softmax;
import com.alibaba.alink.pipeline.dataproc.TensorToVector;
import com.alibaba.alink.pipeline.dataproc.vector.VectorAssembler;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class Chap27 {
    static final String DATA_DIR = Utils.ROOT_DIR + "casia" + File.separator;

    static final String TEST_FILE = "test.ak";
    static final String TRAIN_FILE = "train.ak";
    static final int AUDIO_SAMPLE_RATE = 16000;

    public static void main(String[] args) throws Exception {

        c_1();

        c_2();

        c_3_1();
        c_3_2();

        c_4_1();
        c_4_2();

    }

    static void c_1() throws Exception {
        listWavFiles_Local(DATA_DIR);

        listWavFiles_FileSystem(DATA_DIR);

        new MemSourceBatchOp(listWavFiles_Local(DATA_DIR), "relative_path")
            .select("relative_path, "
                + "REGEXP_EXTRACT(relative_path, '(angry|fear|happy|neutral|sad|surprise)') AS emotion, "
                + "REGEXP_EXTRACT(relative_path, '(liuchanhg|wangzhe|zhaoquanyin|ZhaoZuoxiang)') AS speaker"
            )
            .link(
                new AkSinkBatchOp()
                    .setFilePath(DATA_DIR + "temp.ak")
                    .setOverwriteSink(true)
            );
        BatchOperator.execute();

        BatchOperator<?> data_set = new AkSourceBatchOp()
            .setFilePath(DATA_DIR + "temp.ak")
            .link(
                new ReadAudioToTensorBatchOp()
                    .setRelativeFilePathCol("relative_path")
                    .setRootFilePath(DATA_DIR)
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setDuration(3.0)
                    .setOutputCol("audio_data")
            );

        Utils.splitTrainTestIfNotExist(
            data_set, DATA_DIR + TRAIN_FILE, DATA_DIR + TEST_FILE, 0.9
        );
    }

    static String[] listWavFiles_Local(String dirPath) throws Exception {
        List<String> paths = new ArrayList<>();

        subListWavFiles(new File(dirPath), paths);

        List<String> relativePaths = paths.stream()
            .map(x -> x.substring(dirPath.length()))
            .collect(Collectors.toList());

        for (int i = 0; i < 5; i++) {
            System.out.println(relativePaths.get(i));
        }

        return relativePaths.toArray(new String[0]);
    }

    static private void subListWavFiles(File dir, List<String> relativePaths) throws IOException {
        for (File file : dir.listFiles()) {
            if (file.isDirectory()) {
                subListWavFiles(file, relativePaths);
            } else {
                if (file.getName().toLowerCase().endsWith(".wav")) {
                    relativePaths.add(file.getCanonicalPath());
                }
            }
        }
    }

    static String[] listWavFiles_FileSystem(String dirPath) throws Exception {
        LocalFileSystem fs = new LocalFileSystem();
        FilePath rootFolder = new FilePath(dirPath, fs);
        URI rootUri = rootFolder.getPath().makeQualified(fs).toUri();
        List<String> relativePaths = FileSystemUtils.listFilesRecursive(rootFolder, true)
            .stream()
            .filter(new Predicate<FilePath>() {
                @Override
                public boolean test(FilePath filePath) {
                    return filePath.getPathStr().toLowerCase().endsWith(".wav");
                }
            })
            .map(x -> rootUri
                .relativize(x.getPath().makeQualified(fs).toUri())
                .getPath()
            )
            .collect(Collectors.toList());

        for (int i = 0; i < 5; i++) {
            System.out.println(relativePaths.get(i));
        }

        return relativePaths.toArray(new String[0]);
    }

    static void c_2() throws Exception {
        new AkSourceBatchOp()
            .setFilePath(DATA_DIR + TEST_FILE)
            .sampleWithSize(1)
            .link(
                new ExtractMfccFeatureBatchOp()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
            )
            .print();
    }

    static void c_3_1() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        emotion_softmax_1(train_set, test_set);

        emotion_softmax_2(train_set, test_set);

        emotion_softmax_3(train_set, test_set);

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void emotion_softmax_1(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("emotion")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.FLATTEN)
                    .setOutputCol("mfcc")
            )
            .add(
                new Softmax()
                    .setVectorCol("mfcc")
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .enableLazyPrintModelInfo()
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
    }

    static void emotion_softmax_2(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("emotion")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MEAN)
                    .setOutputCol("mfcc")
            )
            .add(
                new Softmax()
                    .setVectorCol("mfcc")
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .enableLazyPrintModelInfo()
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
    }

    static void emotion_softmax_3(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("emotion")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MEAN)
                    .setOutputCol("mfcc_mean")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MIN)
                    .setOutputCol("mfcc_min")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MAX)
                    .setOutputCol("mfcc_max")
            )
            .add(
                new VectorAssembler()
                    .setSelectedCols("mfcc_mean", "mfcc_min", "mfcc_max")
                    .setOutputCol("mfcc")
            )
            .add(
                new Softmax()
                    .setVectorCol("mfcc")
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .enableLazyPrintModelInfo()
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
    }

    static void c_3_2() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        emotion_cnn(train_set, test_set);

        emotion_cnn2D(train_set, test_set);

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void emotion_cnn(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        BatchOperator.setParallelism(1);

        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("emotion")
                    .setNumThreads(12)
            )
            .add(
                new KerasSequentialClassifier()
                    .setTensorCol("mfcc")
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .setLayers(
                        "Reshape((90, 128))",
                        "Conv1D(256, 5, padding='same', activation='relu')",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Dropout(0.1)",
                        "MaxPooling1D(pool_size=8)",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Flatten()"
                    )
                    .setOptimizer("Adam(lr=0.001,decay=4e-5)")
                    .setBatchSize(32)
                    .setIntraOpParallelism(1)
                    .setNumEpochs(50)
                    .setSaveCheckpointsEpochs(3.0)
                    .setValidationSplit(0.1)
                    .setSaveBestOnly(true)
                    .setBestMetric("sparse_categorical_accuracy")
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );

        BatchOperator.execute();
    }

    static void emotion_cnn2D(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        BatchOperator.setParallelism(1);

        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("emotion")
                    .setNumThreads(12)
            )
            .add(
                new KerasSequentialClassifier()
                    .setTensorCol("mfcc")
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .setLayers(
                        "Conv2D(32, (3,3), padding='same', activation='relu')",
                        "Conv2D(32, (3,3), padding='same', activation='relu')",
                        "AveragePooling2D(pool_size=(3, 3), padding='same')",
                        "Conv2D(64, (3,3), padding='same', strides=(3, 3), activation='relu')",
                        "Conv2D(64, (3,3), padding='same', strides=(3, 3), activation='relu')",
                        "Conv2D(64, (3,3), padding='same', strides=(3, 3), activation='relu')",
                        "AveragePooling2D(pool_size=(3, 3), padding='same')",
                        "Conv2D(128, (3,3), padding='same', strides=(3, 3), activation='relu')",
                        "Flatten()"
                    )
                    .setBatchSize(64)
                    .setIntraOpParallelism(1)
                    .setNumEpochs(50)
                    .setSaveCheckpointsEpochs(3.0)
                    .setValidationSplit(0.1)
                    .setSaveBestOnly(true)
                    .setBestMetric("sparse_categorical_accuracy")
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("emotion")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );

        BatchOperator.execute();
    }

    static void c_4_1() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        speaker_softmax(train_set, test_set);

        speaker_softmax_2(train_set, test_set);

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void speaker_softmax(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("speaker")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MEAN)
                    .setOutputCol("mfcc")
            )
            .add(
                new Softmax()
                    .setVectorCol("mfcc")
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
    }

    static void speaker_softmax_2(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("speaker")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MEAN)
                    .setOutputCol("mfcc_mean")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MIN)
                    .setOutputCol("mfcc_min")
            )
            .add(
                new TensorToVector()
                    .setSelectedCol("mfcc")
                    .setConvertMethod(ConvertMethod.MAX)
                    .setOutputCol("mfcc_max")
            )
            .add(
                new VectorAssembler()
                    .setSelectedCols("mfcc_mean", "mfcc_min", "mfcc_max")
                    .setOutputCol("mfcc")
            )
            .add(
                new Softmax()
                    .setVectorCol("mfcc")
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
    }

    static void c_4_2() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        speaker_cnn(train_set, test_set);

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void speaker_cnn(BatchOperator<?> train_set, BatchOperator<?> test_set) throws Exception {
        BatchOperator.setParallelism(1);

        new Pipeline()
            .add(
                new ExtractMfccFeature()
                    .setSelectedCol("audio_data")
                    .setSampleRate(AUDIO_SAMPLE_RATE)
                    .setOutputCol("mfcc")
                    .setReservedCols("speaker")
                    .setNumThreads(12)
            )
            .add(
                new KerasSequentialClassifier()
                    .setTensorCol("mfcc")
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
                    .setLayers(
                        "Reshape((90, 128))",
                        "Conv1D(256, 5, padding='same', activation='relu')",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Dropout(0.1)",
                        "MaxPooling1D(pool_size=8)",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Conv1D(128, 5, padding='same', activation='relu')",
                        "Flatten()"
                    )
                    .setNumEpochs(50)
                    .setSaveCheckpointsEpochs(3.0)
                    .setValidationSplit(0.1)
                    .setSaveBestOnly(true)
                    .setBestMetric("sparse_categorical_accuracy")
            )
            .fit(train_set)
            .transform(test_set)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("speaker")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics()
            );

        BatchOperator.execute();
    }


}