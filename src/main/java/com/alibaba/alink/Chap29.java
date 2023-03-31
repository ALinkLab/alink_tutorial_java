package com.alibaba.alink;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.utils.TableUtil;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.LogisticRegressionPredictBatchOp;
import com.alibaba.alink.operator.batch.classification.LogisticRegressionTrainBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.sink.AkSinkBatchOp;
import com.alibaba.alink.operator.batch.sink.AppendModelStreamFileSinkBatchOp;
import com.alibaba.alink.operator.batch.source.AkSourceBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.common.evaluation.BinaryClassMetrics;
import com.alibaba.alink.operator.stream.StreamOperator;
import com.alibaba.alink.operator.stream.classification.LogisticRegressionPredictStreamOp;
import com.alibaba.alink.operator.stream.dataproc.JsonValueStreamOp;
import com.alibaba.alink.operator.stream.dataproc.SplitStreamOp;
import com.alibaba.alink.operator.stream.evaluation.EvalBinaryClassStreamOp;
import com.alibaba.alink.operator.stream.onlinelearning.FtrlModelFilterStreamOp;
import com.alibaba.alink.operator.stream.onlinelearning.FtrlPredictStreamOp;
import com.alibaba.alink.operator.stream.onlinelearning.FtrlTrainStreamOp;
import com.alibaba.alink.operator.stream.sink.ModelStreamFileSinkStreamOp;
import com.alibaba.alink.operator.stream.source.CsvSourceStreamOp;
import com.alibaba.alink.operator.stream.source.ModelStreamFileSourceStreamOp;
import com.alibaba.alink.params.feature.HasEncodeWithoutWoe;
import com.alibaba.alink.pipeline.LocalPredictor;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.alink.pipeline.classification.LogisticRegression;
import com.alibaba.alink.pipeline.classification.LogisticRegressionModel;
import com.alibaba.alink.pipeline.dataproc.StandardScaler;
import com.alibaba.alink.pipeline.dataproc.vector.VectorAssembler;
import com.alibaba.alink.pipeline.feature.OneHotEncoder;
import org.apache.commons.lang3.ArrayUtils;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Chap29 {

    private static final String DATA_DIR = Utils.ROOT_DIR + "ctr_avazu" + File.separator;

    static final String SCHEMA_STRING
        = "id string, click string, dt string, C1 string, banner_pos int, site_id string, site_domain string, "
        + "site_category string, app_id string, app_domain string, app_category string, device_id string, "
        + "device_ip string, device_model string, device_type string, device_conn_type string, C14 int, C15 int, "
        + "C16 int, C17 int, C18 int, C19 int, C20 int, C21 int";

    static final String[] CATEGORY_COL_NAMES = new String[]{
        "C1", "banner_pos", "site_category", "app_domain",
        "app_category", "device_type", "device_conn_type",
        "site_id", "site_domain", "device_id", "device_model"};

    static final String[] NUMERICAL_COL_NAMES = new String[]{
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"};

    static final String INIT_NUMERIC_LR_MODEL_FILE = "init_numeric_lr_model.ak";
    static final String FTRL_MODEL_STREAM_DIR = "ftrl_model_stream";
    static final String LR_PIPELINEMODEL_FILE = "lr_pipelinemodel.ak";
    static final String INIT_PIPELINE_MODEL_FILE = "init_pipeline_model.ak";
    static final String PIPELINE_MODEL_STREAM_DIR = "pipeline_model_stream";
    static final String PIPELINEMODEL_WITH_MODELSTREAM_FILE = "pipelinemodel_with_modelstream.ak";
    static final String FILTERED_MODEL_STREAM_DIR = "filtered_model_stream";

    static final String LABEL_COL_NAME = "click";
    static final String VEC_COL_NAME = "vec";
    static final String PREDICTION_COL_NAME = "pred";
    static final String PRED_DETAIL_COL_NAME = "pred_info";

    public static void main(String[] args) throws Exception {

        c_1_1();
        c_1_2();

        c_2();

        c_3_1();
        c_3_2();
        c_3_3();

        c_4_1();
        c_4_2();
        c_4_3();

        c_5();

        c_6_1();
        c_6_2();

    }

    static void c_1_1() throws Exception {

        if (!new File(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE).exists()) {
            new CsvSourceBatchOp()
                .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-small.csv")
                .setSchemaStr(SCHEMA_STRING)
                .link(
                    new LogisticRegressionTrainBatchOp()
                        .setFeatureCols(NUMERICAL_COL_NAMES)
                        .setLabelCol(LABEL_COL_NAME)
                        .setMaxIter(10)
                )
                .link(
                    new AkSinkBatchOp()
                        .setFilePath(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE)
                );
            BatchOperator.execute();
        }

        AkSourceBatchOp initModel = new AkSourceBatchOp()
            .setFilePath(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE);

        // prepare stream train data
        CsvSourceStreamOp data = new CsvSourceStreamOp()
            .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
            .setSchemaStr(SCHEMA_STRING)
            .setIgnoreFirstLine(true);

        // split stream to train and eval data
        SplitStreamOp spliter = new SplitStreamOp().setFraction(0.5).linkFrom(data);
        StreamOperator train_stream_data = spliter;
        StreamOperator test_stream_data = spliter.getSideOutput(0);

        // ftrl train
        FtrlTrainStreamOp model_stream = new FtrlTrainStreamOp(initModel)
            .setFeatureCols(NUMERICAL_COL_NAMES)
            .setLabelCol(LABEL_COL_NAME)
            .setTimeInterval(10)
            .linkFrom(train_stream_data);

        // ftrl predict
        FtrlPredictStreamOp predResult = new FtrlPredictStreamOp(initModel)
            .setPredictionCol(PREDICTION_COL_NAME)
            .setReservedCols(LABEL_COL_NAME)
            .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
            .linkFrom(model_stream, test_stream_data);

        predResult
            .sample(0.0001)
            .select("'Pred Sample' AS out_type, *")
            .print();

        // ftrl eval
        predResult
            .link(
                new EvalBinaryClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setTimeInterval(10)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.AUC", "$.ConfusionMatrix"})
            )
            .select("'Eval Metric' AS out_type, *")
            .print();

        StreamOperator.execute();

    }

    static void c_1_2() throws Exception {
        AkSourceBatchOp initModel = new AkSourceBatchOp()
            .setFilePath(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE);

        // prepare stream train data
        CsvSourceStreamOp data = new CsvSourceStreamOp()
            .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
            .setSchemaStr(SCHEMA_STRING)
            .setIgnoreFirstLine(true);

        // split stream to train and eval data
        SplitStreamOp spliter = new SplitStreamOp().setFraction(0.5).linkFrom(data);
        StreamOperator train_stream_data = spliter;
        StreamOperator test_stream_data = spliter.getSideOutput(0);

        // ftrl train
        FtrlTrainStreamOp model_stream = new FtrlTrainStreamOp(initModel)
            .setFeatureCols(NUMERICAL_COL_NAMES)
            .setLabelCol(LABEL_COL_NAME)
            .setTimeInterval(10)
            .linkFrom(train_stream_data);

        model_stream.link(
            new ModelStreamFileSinkStreamOp()
                .setFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
                .setNumKeepModel(5)
        );

        StreamOperator<?> new_model_stream = new ModelStreamFileSourceStreamOp()
            .setSchemaStr(TableUtil.schema2SchemaStr(initModel.getSchema()))
            .setFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR);

        // ftrl predict
        FtrlPredictStreamOp predResult = new FtrlPredictStreamOp(initModel)
            .setPredictionCol(PREDICTION_COL_NAME)
            .setReservedCols(LABEL_COL_NAME)
            .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
            .linkFrom(new_model_stream, test_stream_data);
        predResult
            .sample(0.0001)
            .select("'Pred Sample' AS out_type, *")
            .print();

        // ftrl eval
        predResult
            .link(
                new EvalBinaryClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setTimeInterval(10)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.AUC", "$.ConfusionMatrix"})
            )
            .select("'Eval Metric' AS out_type, *")
            .print();

        StreamOperator.execute();
    }

    static void c_2() throws Exception {
        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        for (int i = 1; i <= 10; i++) {

            new CsvSourceBatchOp()
                .setFilePath(DATA_DIR + "avazu-small.csv")
                .setSchemaStr(SCHEMA_STRING)
                .sample(0.1 * i)
                .link(
                    new LogisticRegressionTrainBatchOp()
                        .setFeatureCols(NUMERICAL_COL_NAMES)
                        .setLabelCol(LABEL_COL_NAME)
                )
                .link(
                    new AppendModelStreamFileSinkBatchOp()
                        .setFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
                        .setNumKeepModel(10)
                );
            BatchOperator.execute();

            System.out.println("\nTrain " + String.valueOf(i) + " models.\n");

            Thread.sleep(2000);
        }
    }

    static void c_3_1() throws Exception {
        BatchOperator initModel = new AkSourceBatchOp()
            .setFilePath(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE);

        StreamOperator<?> predResult = new CsvSourceStreamOp()
            .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
            .setSchemaStr(SCHEMA_STRING)
            .setIgnoreFirstLine(true)
            .link(
                new LogisticRegressionPredictStreamOp(initModel)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .setReservedCols(new String[]{LABEL_COL_NAME})
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setModelStreamFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
            );

        predResult
            .sample(0.0001)
            .select("'Pred Sample' AS out_type, *")
            .print();

        predResult
            .link(
                new EvalBinaryClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setTimeInterval(10)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.AUC", "$.ConfusionMatrix"})
            )
            .select("'Eval Metric' AS out_type, *")
            .print();

        StreamOperator.execute();
    }

    static void c_3_2() throws Exception {
        BatchOperator initModel = new AkSourceBatchOp()
            .setFilePath(DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE);

        PipelineModel pipelineModel = new PipelineModel(
            new LogisticRegressionModel()
                .setModelData(initModel)
                .setPredictionCol(PREDICTION_COL_NAME)
                .setReservedCols(new String[]{LABEL_COL_NAME})
                .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                .setModelStreamFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
        );

        pipelineModel.save(DATA_DIR + LR_PIPELINEMODEL_FILE, true);
        BatchOperator.execute();

        StreamOperator<?> predResult = pipelineModel
            .transform(
                new CsvSourceStreamOp()
                    .setFilePath(
                        "http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
                    .setSchemaStr(SCHEMA_STRING)
                    .setIgnoreFirstLine(true)
            );

        predResult
            .sample(0.0001)
            .select("'Pred Sample' AS out_type, *")
            .print();

        predResult
            .link(
                new EvalBinaryClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setTimeInterval(10)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                    .setJsonPath(new String[]{"$.Accuracy", "$.AUC", "$.ConfusionMatrix"})
            )
            .select("'Eval Metric' AS out_type, *")
            .print();

        StreamOperator.execute();
    }

    static void c_3_3() throws Exception {
        Object[] input = new Object[]{
            "10000949271186029916", "1", "14102100", "1005", 0, "1fbe01fe", "f3845767", "28905ebd",
            "ecad2386", "7801e8d9", "07d7df22", "a99f214a", "37e8da74", "5db079b5", "1", "2",
            15707, 320, 50, 1722, 0, 35, -1, 79};

        LocalPredictor localPredictor
            = new LocalPredictor(DATA_DIR + LR_PIPELINEMODEL_FILE, SCHEMA_STRING);

        for (int i = 1; i <= 100; i++) {
            System.out.print(i + "\t");
            System.out.println(ArrayUtils.toString(localPredictor.predict(input)));
            Thread.sleep(2000);
        }
        localPredictor.close();
    }

    static void c_4_1() throws Exception {
        Pipeline pipeline = new Pipeline()
            .add(
                new StandardScaler()
                    .setSelectedCols(NUMERICAL_COL_NAMES)
            )
            .add(
                new OneHotEncoder()
                    .setSelectedCols(CATEGORY_COL_NAMES)
                    .setDropLast(false)
                    .setEncode(HasEncodeWithoutWoe.Encode.ASSEMBLED_VECTOR)
                    .setOutputCols(VEC_COL_NAME)
            )
            .add(
                new VectorAssembler()
                    .setSelectedCols(ArrayUtils.add(NUMERICAL_COL_NAMES, VEC_COL_NAME))
                    .setOutputCol(VEC_COL_NAME)
            )
            .add(
                new LogisticRegression()
                    .setVectorCol(VEC_COL_NAME)
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .setReservedCols(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
            );

        CsvSourceBatchOp train_set = new CsvSourceBatchOp()
            .setFilePath(DATA_DIR + "avazu-small.csv")
            .setSchemaStr(SCHEMA_STRING);

        pipeline
            .fit(train_set.sample(0.05))
            .save(DATA_DIR + INIT_PIPELINE_MODEL_FILE, true);

        BatchOperator.execute();

        for (int i = 2; i <= 20; i++) {
            pipeline
                .fit(train_set.sample(0.05 * i))
                .save()
                .link(
                    new AppendModelStreamFileSinkBatchOp()
                        .setFilePath(DATA_DIR + PIPELINE_MODEL_STREAM_DIR)
                        .setNumKeepModel(19)
                );

            BatchOperator.execute();

            System.out.println("\nTrain " + (i - 1) + " PipelineModels.\n");

            Thread.sleep(2000);
        }
    }

    static void c_4_2() throws Exception {
        StreamOperator<?> predResult = PipelineModel
            .load(DATA_DIR + INIT_PIPELINE_MODEL_FILE)
            .setModelStreamFilePath(DATA_DIR + PIPELINE_MODEL_STREAM_DIR)
            .transform(
                new CsvSourceStreamOp()
                    .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
                    .setSchemaStr(SCHEMA_STRING)
                    .setIgnoreFirstLine(true)
            );

        predResult
            .sample(0.0001)
            .select("'Pred Sample' AS out_type, *")
            .print();

        predResult
            .link(
                new EvalBinaryClassStreamOp()
                    .setLabelCol(LABEL_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .setTimeInterval(10)
            )
            .link(
                new JsonValueStreamOp()
                    .setSelectedCol("Data")
                    .setReservedCols(new String[]{"Statistics"})
                    .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                    .setJsonPath("$.Accuracy", "$.AUC", "$.ConfusionMatrix")
            )
            .select("'Eval Metric' AS out_type, *")
            .print();

        StreamOperator.execute();
    }

    static void c_4_3() throws Exception {
        PipelineModel
            .load(DATA_DIR + INIT_PIPELINE_MODEL_FILE)
            .setModelStreamFilePath(DATA_DIR + PIPELINE_MODEL_STREAM_DIR)
            .save(DATA_DIR + PIPELINEMODEL_WITH_MODELSTREAM_FILE, true);
        BatchOperator.execute();

        LocalPredictor localPredictor
            = new LocalPredictor(DATA_DIR + PIPELINEMODEL_WITH_MODELSTREAM_FILE, SCHEMA_STRING);

        Object[] input = new Object[]{
            "10000949271186029916", "1", "14102100", "1005", 0, "1fbe01fe", "f3845767", "28905ebd",
            "ecad2386", "7801e8d9", "07d7df22", "a99f214a", "37e8da74", "5db079b5", "1", "2",
            15707, 320, 50, 1722, 0, 35, -1, 79};

        for (int i = 1; i <= 100; i++) {
            System.out.print(i + "\t");
            System.out.println(ArrayUtils.toString(localPredictor.predict(input)));
            Thread.sleep(2000);
        }

        localPredictor.close();
    }

    static void c_5() throws Exception {
        String current_base_model_path = DATA_DIR + INIT_NUMERIC_LR_MODEL_FILE;

        for (int i = 0; i < 10; i++) {

            if (i > 0) {
                long latest_time = -1L;
                for (File subdir : new File(DATA_DIR + FTRL_MODEL_STREAM_DIR).listFiles()) {
                    if (!subdir.getName().equals("conf") && subdir.lastModified() > latest_time) {
                        latest_time = subdir.lastModified();
                        current_base_model_path = subdir.getCanonicalPath();
                    }
                }
            }

            System.out.println(current_base_model_path);

            BatchOperator<?> train_set = new CsvSourceBatchOp()
                .setFilePath(DATA_DIR + "avazu-small.csv")
                .setSchemaStr(SCHEMA_STRING);

            AkSourceBatchOp init_model = new AkSourceBatchOp().setFilePath(current_base_model_path);

            new LogisticRegressionTrainBatchOp()
                .setFeatureCols(NUMERICAL_COL_NAMES)
                .setLabelCol(LABEL_COL_NAME)
                .setMaxIter(5)
                .linkFrom(train_set, init_model)
                .link(
                    new AppendModelStreamFileSinkBatchOp()
                        .setFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
                        .setNumKeepModel(10)
                );
            BatchOperator.execute();

            Thread.sleep(2000);
        }
    }

    static void c_6_1() throws Exception {
        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        StreamOperator<?> source_model_stream = new ModelStreamFileSourceStreamOp()
            .setFilePath(DATA_DIR + FTRL_MODEL_STREAM_DIR)
            .setStartTime("2021-01-01 00:00:00");

        CsvSourceStreamOp val_stream_data = new CsvSourceStreamOp()
            .setFilePath("http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-ctr-train-8M.csv")
            .setSchemaStr(SCHEMA_STRING)
            .setIgnoreFirstLine(true);

        FtrlModelFilterStreamOp model_filter = new FtrlModelFilterStreamOp()
            .setPositiveLabelValueString("1")
            .setLabelCol(LABEL_COL_NAME)
            .setAccuracyThreshold(0.8)
            .setAucThreshold(0.6);

        model_filter.linkFrom(source_model_stream, val_stream_data);

        model_filter
            .link(
                new ModelStreamFileSinkStreamOp()
                    .setFilePath(DATA_DIR + FILTERED_MODEL_STREAM_DIR)
                    .setNumKeepModel(10)
            );

        StreamOperator.execute();
    }

    static void c_6_2() throws Exception {
        List<File> model_dirs = Arrays.asList(
            new File(DATA_DIR + FTRL_MODEL_STREAM_DIR)
                .listFiles(
                    new FilenameFilter() {
                        @Override
                        public boolean accept(File dir, String name) {
                            return name.length() > 10;
                        }
                    }
                )
        );

        Collections.sort(model_dirs,
            new Comparator<File>() {
                @Override
                public int compare(File o1, File o2) {
                    return o1.getName().compareTo(o2.getName());
                }
            }
        );

        for (File model_dir : model_dirs) {

            CsvSourceBatchOp validation_set = new CsvSourceBatchOp()
                .setFilePath(DATA_DIR + "avazu-small.csv")
                .setSchemaStr(SCHEMA_STRING);

            AkSourceBatchOp model = new AkSourceBatchOp().setFilePath(model_dir.getCanonicalPath());

            BinaryClassMetrics metrics =
                new LogisticRegressionPredictBatchOp()
                    .setPredictionCol(PREDICTION_COL_NAME)
                    .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    .linkFrom(model, validation_set)
                    .link(
                        new EvalBinaryClassBatchOp()
                            .setPositiveLabelValueString("1")
                            .setLabelCol(LABEL_COL_NAME)
                            .setPredictionDetailCol(PRED_DETAIL_COL_NAME)
                    )
                    .collectMetrics();

            System.out.println(model_dir.getName());
            System.out.println("auc : " + metrics.getAuc() + ",\t accuracy : " + metrics.getAccuracy());

            if (metrics.getAuc() > 0.6 && metrics.getAccuracy() > 0.8) {
                model.link(
                    new AppendModelStreamFileSinkBatchOp()
                        .setFilePath(DATA_DIR + FILTERED_MODEL_STREAM_DIR)
                        .setNumKeepModel(10)
                );
                BatchOperator.execute();
            }

        }

    }

}