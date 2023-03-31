package com.alibaba.alink;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.common.utils.Stopwatch;
import com.alibaba.alink.common.utils.TableUtil;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.source.AkSourceBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.pipeline.LocalPredictor;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.alink.pipeline.classification.BertTextClassifier;
import com.alibaba.alink.pipeline.classification.LogisticRegression;
import com.alibaba.alink.pipeline.dataproc.Imputer;
import com.alibaba.alink.pipeline.nlp.BertTextEmbedding;
import com.alibaba.alink.pipeline.nlp.DocCountVectorizer;
import com.alibaba.alink.pipeline.nlp.Segment;
import com.alibaba.alink.pipeline.nlp.StopWordsRemover;

import java.io.File;

public class Chap28 {
    static final String DATA_DIR = Utils.ROOT_DIR + "sentiment_hotel" + File.separator;
    static final String ORIGIN_FILE = "ChnSentiCorp_htl_all.csv";
    static final String TEST_FILE = "test.ak";
    static final String TRAIN_FILE = "train.ak";

    public static void main(String[] args) throws Exception {

        c_1_1();
        c_1_2();

        c_2();

        c_3();

    }

    static void c_1_1() throws Exception {
        CsvSourceBatchOp source = new CsvSourceBatchOp()
            .setFilePath(DATA_DIR + ORIGIN_FILE)
            .setSchemaStr("label int, review string")
            .setIgnoreFirstLine(true);

        Utils.splitTrainTestIfNotExist(source, DATA_DIR + TRAIN_FILE, DATA_DIR + TEST_FILE, 0.9);

        source.firstN(5).print();
    }

    static void c_1_2() throws Exception {
        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        new Pipeline()
            .add(
                new Imputer()
                    .setSelectedCols("review")
                    .setOutputCols("featureText")
                    .setStrategy("value")
                    .setFillValue("null")
            )
            .add(
                new Segment()
                    .setSelectedCol("featureText")
            )
            .add(
                new StopWordsRemover()
                    .setSelectedCol("featureText")
            )
            .add(
                new DocCountVectorizer()
                    .setFeatureType("TF")
                    .setSelectedCol("featureText")
                    .setOutputCol("featureVector")
            )
            .add(
                new LogisticRegression()
                    .setVectorCol("featureVector")
                    .setLabelCol("label")
                    .setPredictionCol("pred")
                    .setPredictionDetailCol("pred_info")
            )
            .fit(train_set)
            .save(DATA_DIR + "lr_pipeline_model.ak", true);

        BatchOperator.execute();

        PipelineModel.load(DATA_DIR + "lr_pipeline_model.ak")
            .transform(test_set)
            .lazyPrint(5)
            .link(
                new EvalBinaryClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionDetailCol("pred_info")
                    .lazyPrintMetrics("LR")
            );
        BatchOperator.execute();

        PipelineModel.load(DATA_DIR + "lr_pipeline_model.ak")
            .transform(test_set)
            .lazyPrint(5)
            .link(
                new EvalBinaryClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionDetailCol("pred_info")
                    .lazyPrintMetrics("LR")
            );
        BatchOperator.execute();

        LocalPredictor localPredictor
            = new LocalPredictor(DATA_DIR + "lr_pipeline_model.ak", "review string");

        final int index_pred = TableUtil.findColIndex(localPredictor.getOutputSchema(), "pred");

        String[] reviews = new String[]{
            "硬件不错，服务态度也不错，下次到附近的话还会选择住这里",
            "房间还比较干净,交通方便,离外滩很近.但外面声音太大,休息不好",
        };

        for (String review : reviews) {
            Object[] result = localPredictor.predict(review);
            System.out.println("Pred Result : " + result[index_pred] + " @ " + review);
        }

    }

    static void c_2() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        if (!new File(DATA_DIR + "bert_vec_lr_pipeline_model.ak").exists()) {
            new Pipeline()
                .add(
                    new Imputer()
                        .setSelectedCols("review")
                        .setStrategy("value")
                        .setFillValue("null")
                )
                .add(
                    new BertTextEmbedding()
                        .setBertModelName("Base-Chinese")
                        .setSelectedCol("review")
                        .setOutputCol("vec")
                )
                .add(
                    new LogisticRegression()
                        .setVectorCol("vec")
                        .setLabelCol("label")
                        .setPredictionCol("pred")
                        .setPredictionDetailCol("pred_info")
                )
                .fit(train_set)
                .save(DATA_DIR + "bert_vec_lr_pipeline_model.ak", true);
            BatchOperator.execute();
        }

        PipelineModel.load(DATA_DIR + "bert_vec_lr_pipeline_model.ak")
            .transform(test_set)
            .lazyPrint(5)
            .link(
                new EvalBinaryClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionDetailCol("pred_info")
                    .lazyPrintMetrics("BERT_VEC_LR")
            );
        BatchOperator.execute();

        LocalPredictor localPredictor
            = new LocalPredictor(DATA_DIR + "bert_vec_lr_pipeline_model.ak", "review string");

        System.out.println(localPredictor.getOutputSchema());

        final int index_pred = TableUtil.findColIndex(localPredictor.getOutputSchema(), "pred");

        String[] reviews = new String[]{
            "硬件不错，服务态度也不错，下次到附近的话还会选择住这里",
            "房间还比较干净,交通方便,离外滩很近.但外面声音太大,休息不好",
        };

        for (String review : reviews) {
            Object[] result = localPredictor.predict(review);
            System.out.println("Pred Result : " + result[index_pred] + " @ " + review);
        }

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }

    static void c_3() throws Exception {
        Stopwatch sw = new Stopwatch();
        sw.start();

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        BatchOperator.setParallelism(4);

        AkSourceBatchOp train_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TRAIN_FILE);
        AkSourceBatchOp test_set = new AkSourceBatchOp().setFilePath(DATA_DIR + TEST_FILE);

        if (!new File(DATA_DIR + "bert_pipeline_model.ak").exists()) {
            new Pipeline()
                .add(
                    new Imputer()
                        .setSelectedCols("review")
                        .setStrategy("value")
                        .setFillValue("null")
                )
                .add(
                    new BertTextClassifier()
                        .setTextCol("review")
                        .setLabelCol("label")
                        .setPredictionCol("pred")
                        .setPredictionDetailCol("pred_info")
                        .setBertModelName("Base-Chinese")
                        .setNumEpochs(1.0)
                )
                .fit(train_set)
                .save(DATA_DIR + "bert_pipeline_model.ak", true);
            BatchOperator.execute();
        }

        PipelineModel.load(DATA_DIR + "bert_pipeline_model.ak")
            .transform(test_set)
            .lazyPrint(5)
            .link(
                new EvalBinaryClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionDetailCol("pred_info")
                    .lazyPrintMetrics("BERT")
            );
        BatchOperator.execute();

        LocalPredictor localPredictor
            = new LocalPredictor(DATA_DIR + "bert_pipeline_model.ak", "review string");

        System.out.println(localPredictor.getOutputSchema());

        final int index_pred = TableUtil.findColIndex(localPredictor.getOutputSchema(), "pred");

        String[] reviews = new String[]{
            "硬件不错，服务态度也不错，下次到附近的话还会选择住这里",
            "房间还比较干净,交通方便,离外滩很近.但外面声音太大,休息不好",
        };

        for (String review : reviews) {
            Object[] result = localPredictor.predict(review);
            System.out.println("Pred Result : " + result[index_pred] + " @ " + review);
        }

        sw.stop();
        System.out.println(sw.getElapsedTimeSpan());
    }


}