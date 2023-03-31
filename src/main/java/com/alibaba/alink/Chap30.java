package com.alibaba.alink;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.utils.Stopwatch;
import com.alibaba.alink.common.utils.TableUtil;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.KnnPredictBatchOp;
import com.alibaba.alink.operator.batch.classification.KnnTrainBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalMultiClassBatchOp;
import com.alibaba.alink.operator.batch.sink.AkSinkBatchOp;
import com.alibaba.alink.operator.batch.source.AkSourceBatchOp;
import com.alibaba.alink.operator.stream.StreamOperator;
import com.alibaba.alink.operator.stream.classification.KnnPredictStreamOp;
import com.alibaba.alink.operator.stream.dataproc.JsonValueStreamOp;
import com.alibaba.alink.operator.stream.evaluation.EvalMultiClassStreamOp;
import com.alibaba.alink.operator.stream.source.AkSourceStreamOp;
import com.alibaba.alink.pipeline.LocalPredictor;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.alink.pipeline.classification.KnnClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class Chap30 {

    static final String DATA_DIR = Utils.ROOT_DIR + "mnist" + File.separator;

    static final String SPARSE_TRAIN_FILE = "sparse_train.ak";
    static final String SPARSE_TEST_FILE = "sparse_test.ak";

    static final String KNN_MODEL_FILE = "knn_model.ak";
    static final String PIPELINE_MODEL_FILE = "pipeline_model.ak";

    static final String VECTOR_COL_NAME = "vec";
    static final String LABEL_COL_NAME = "label";
    static final String PREDICTION_COL_NAME = "id_cluster";

    public static void main(String[] args) throws Exception {

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        c_0();

        c_1_1();
        c_1_2();

        c_2_1();
        c_2_2();
        c_2_3();

        c_3();

    }

    static BatchOperator<?> getTrainSet() {
        return new AkSourceBatchOp().setFilePath(DATA_DIR + SPARSE_TRAIN_FILE);
    }

    static BatchOperator<?> getTestSet() {
        return new AkSourceBatchOp().setFilePath(DATA_DIR + SPARSE_TEST_FILE);
    }

    static StreamOperator<?> getTestStream() {
        return new AkSourceStreamOp().setFilePath(DATA_DIR + SPARSE_TEST_FILE);
    }

    static void c_0() throws Exception {
        getTrainSet()
            .link(
                new KnnTrainBatchOp()
                    .setVectorCol(VECTOR_COL_NAME)
                    .setLabelCol(LABEL_COL_NAME)
            )
            .link(
                new AkSinkBatchOp()
                    .setFilePath(DATA_DIR + KNN_MODEL_FILE)
                    .setOverwriteSink(true)
            );
        BatchOperator.execute();

        new Pipeline()
            .add(
                new KnnClassifier()
                    .setK(3)
                    .setVectorCol(VECTOR_COL_NAME)
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .fit(getTrainSet())
            .save(DATA_DIR + PIPELINE_MODEL_FILE, true);
        BatchOperator.execute();
    }

    static BatchOperator<?> getKnnModel() {
        return new AkSourceBatchOp().setFilePath(DATA_DIR + KNN_MODEL_FILE);
    }

    static PipelineModel getPipelineModel() {
        return PipelineModel.load(DATA_DIR + PIPELINE_MODEL_FILE);
    }

    static void c_1_1() throws Exception {
        BatchOperator.setParallelism(4);

        Stopwatch sw = new Stopwatch();
        System.out.println("Batch mode with Parallelism=4");
        sw.reset();
        sw.start();
        new KnnPredictBatchOp()
            .setK(3)
            .setVectorCol(VECTOR_COL_NAME)
            .setPredictionCol(PREDICTION_COL_NAME)
            .linkFrom(getKnnModel(), getTestSet())
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());

        System.out.println("Pipeline batch mode with Parallelism=4");
        sw.reset();
        sw.start();
        getPipelineModel()
            .transform(getTestSet())
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_1_2() throws Exception {
        StreamOperator.setParallelism(4);

        Stopwatch sw = new Stopwatch();

        System.out.println("Stream mode with Parallelism=4");
        sw.reset();
        sw.start();
        getTestStream()
            .link(
                new KnnPredictStreamOp(getKnnModel())
                    .setK(3)
                    .setVectorCol(VECTOR_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .link(
                new EvalMultiClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "Kappa"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.Kappa"})
            )
            .print();
        StreamOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());

        System.out.println("Pipeline stream mode with Parallelism=4");
        sw.reset();
        sw.start();
        getPipelineModel()
            .transform(getTestStream())
            .link(
                new EvalMultiClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "Kappa"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.Kappa"})
            )
            .print();
        StreamOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_2_1() throws Exception {
        BatchOperator.setParallelism(4);

        System.out.println("Batch mode with Parallelism=4 and NumThreads=2");
        Stopwatch sw = new Stopwatch();
        sw.reset();
        sw.start();
        new KnnPredictBatchOp()
            .setK(3)
            .setVectorCol(VECTOR_COL_NAME)
            .setPredictionCol(PREDICTION_COL_NAME)
            .setNumThreads(2)
            .linkFrom(getKnnModel(), getTestSet())
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_2_2() throws Exception {
        StreamOperator.setParallelism(4);

        System.out.println("Stream mode with Parallelism=4 and NumThreads=2");
        Stopwatch sw = new Stopwatch();
        sw.reset();
        sw.start();
        getTestStream()
            .link(
                new KnnPredictStreamOp(getKnnModel())
                    .setK(3)
                    .setVectorCol(VECTOR_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .setNumThreads(2)
            )
            .link(
                new EvalMultiClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "Kappa"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.Kappa"})
            )
            .print();
        StreamOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_2_3() throws Exception {
        Stopwatch sw = new Stopwatch();

        BatchOperator.setParallelism(4);
        System.out.println("Pipeline batch mode with Parallelism=4 and NumThreads=2");
        sw.reset();
        sw.start();
        getPipelineModel()
            .setNumThreads(2)
            .transform(getTestSet())
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .lazyPrintMetrics()
            );
        BatchOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());

        StreamOperator.setParallelism(4);
        System.out.println("Pipeline stream mode with Parallelism=4 and NumThreads=2");
        sw.reset();
        sw.start();
        getPipelineModel()
            .setNumThreads(2)
            .transform(getTestStream())
            .link(
                new EvalMultiClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "Kappa"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.Kappa"})
            )
            .print();
        StreamOperator.execute();
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_3() throws Exception {
        String recStr = "$784$129:57.0 130:201.0 131:229.0 132:31.0 157:100.0 158:252.0 159:252.0 160:55.0 185:100.0 "
            + "186:252.0 187:252.0 188:55.0 212:6.0 213:209.0 214:252.0 215:247.0 216:50.0 240:138.0 241:252.0 "
            + "242:252.0 243:173.0 267:65.0 268:236.0 269:252.0 270:235.0 271:19.0 295:244.0 296:252.0 297:252.0 "
            + "298:77.0 322:20.0 323:253.0 324:252.0 325:192.0 326:4.0 350:111.0 351:253.0 352:252.0 353:120.0 "
            + "377:34.0 378:220.0 379:253.0 380:223.0 381:25.0 405:93.0 406:253.0 407:255.0 408:125.0 432:41.0 "
            + "433:204.0 434:252.0 435:230.0 436:23.0 460:154.0 461:252.0 462:252.0 463:177.0 487:127.0 488:248.0 "
            + "489:252.0 490:243.0 491:5.0 514:20.0 515:236.0 516:252.0 517:235.0 518:64.0 541:20.0 542:193.0 "
            + "543:252.0 544:252.0 545:89.0 569:56.0 570:252.0 571:252.0 572:252.0 573:70.0 597:123.0 598:252.0 "
            + "599:252.0 600:245.0 601:97.0 625:165.0 626:252.0 627:252.0 628:127.0 653:70.0 654:252.0 655:146.0 "
            + "656:13.0";

        final int N = 10000;

        Stopwatch sw = new Stopwatch();
        long sum;

        LocalPredictor localPredictor =
            new LocalPredictor(DATA_DIR + PIPELINE_MODEL_FILE, VECTOR_COL_NAME + " string");

        System.out.println(localPredictor.getOutputSchema());

        final int index_pred = TableUtil.findColIndex(localPredictor.getOutputSchema(), PREDICTION_COL_NAME);

        sw.reset();
        sw.start();
        sum = 0;
        for (int i = 0; i < N; i++) {
            sum += (Integer) localPredictor.predict(recStr)[index_pred];
        }
        sw.stop();
        System.out.println(sum);
        System.out.println(sw.getElapsedTimeSpan());

        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
            4, 4, 0, TimeUnit.SECONDS,
            new ArrayBlockingQueue<Runnable>(50),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );

        sw.reset();
        sw.start();
        for (int i = 0; i < N; i++) {
            threadPoolExecutor.submit(new MyRunnableTask(localPredictor, recStr));
        }
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());

        sw.reset();
        sw.start();
        sum = 0;
        int K = 1000;
        ArrayList <MyCallableTask> tasks = new ArrayList <>(K);
        for (int i = 0; i < N / K; i++) {
            tasks.clear();
            for (int k = 0; k < K; k++) {
                tasks.add(new MyCallableTask(localPredictor, recStr));
            }
            List<Future<Object[]>> futures = threadPoolExecutor.invokeAll(tasks);
            for (Future <Object[]> future : futures) {
                sum += (Integer) future.get()[index_pred];
            }
        }
        System.out.println(sum);
        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());

        threadPoolExecutor.shutdown();
    }

    public static class MyRunnableTask implements Runnable {
        private LocalPredictor localPredictor;
        private String taskData;

        public MyRunnableTask(LocalPredictor localPredictor, String taskData) {
            this.localPredictor = localPredictor;
            this.taskData = taskData;
        }

        @Override
        public void run() {
            try {
                localPredictor.predict(taskData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static class MyCallableTask implements Callable<Object[]> {
        private LocalPredictor localPredictor;
        private String taskData;

        public MyCallableTask(LocalPredictor localPredictor, String taskData) {
            this.localPredictor = localPredictor;
            this.taskData = taskData;
        }

        @Override
        public Object[] call() throws Exception {
            return localPredictor.predict(taskData);
        }
    }
}
