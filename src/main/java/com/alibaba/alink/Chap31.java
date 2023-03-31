package com.alibaba.alink;

import com.alibaba.alink.common.AlinkGlobalConfiguration;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.dataproc.LookupBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalMultiClassBatchOp;
import com.alibaba.alink.operator.batch.graph.DeepWalkBatchOp;
import com.alibaba.alink.operator.batch.graph.MetaPath2VecBatchOp;
import com.alibaba.alink.operator.batch.graph.Node2VecBatchOp;
import com.alibaba.alink.operator.batch.sink.AkSinkBatchOp;
import com.alibaba.alink.operator.batch.source.AkSourceBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.batch.source.TextSourceBatchOp;
import com.alibaba.alink.operator.batch.sql.JoinBatchOp;
import com.alibaba.alink.operator.batch.sql.UnionBatchOp;
import com.alibaba.alink.operator.batch.statistics.VectorSummarizerBatchOp;
import com.alibaba.alink.pipeline.classification.KnnClassifier;
import com.alibaba.alink.pipeline.classification.Softmax;

import java.io.File;

public class Chap31 {

    static final String DATA_DIR = Utils.ROOT_DIR + "net_dbis" + File.separator;

    static final String DEEPWALK_EMBEDDING = "deepwalk_embedding.ak";
    static final String NODE2VEC_EMBEDDING = "node2vec_embedding.ak";
    static final String METAPATH2VEC_EMBEDDING = "metapath2vec_embedding.ak";

    static final String AUTHOR_LABEL_TRAIN = "author_label_train.ak";
    static final String AUTHOR_LABEL_TEST = "author_label_test.ak";


    static BatchOperator<?> paper_author = new CsvSourceBatchOp()
        .setFilePath(DATA_DIR + "paper_author.txt")
        .setSchemaStr("paper_id string, author_id string")
        .setFieldDelimiter("\t")
        .select("CONCAT('P', paper_id) AS paper_id, CONCAT('A', author_id) AS author_id");

    static BatchOperator<?> paper_conf = new CsvSourceBatchOp()
        .setFilePath(DATA_DIR + "paper_conf.txt")
        .setSchemaStr("paper_id string, conf_id string")
        .setFieldDelimiter("\t")
        .select("CONCAT('P', paper_id) AS paper_id, CONCAT('C', conf_id) AS conf_id");

    static BatchOperator<?> id_author = new CsvSourceBatchOp()
        .setFilePath(DATA_DIR + "id_author.txt")
        .setSchemaStr("author_id string, author string")
        .setFieldDelimiter("\t")
        .select("CONCAT('A', author_id) AS author_id, author");

    static BatchOperator<?> id_conf = new CsvSourceBatchOp()
        .setFilePath(DATA_DIR + "id_conf.txt")
        .setSchemaStr("conf_id string, conf string")
        .setFieldDelimiter("\t")
        .select("CONCAT('C', conf_id) AS conf_id, conf");

    static BatchOperator<?> paper = new TextSourceBatchOp()
        .setFilePath(DATA_DIR + "paper.txt")
        .select("CONCAT('P', TRIM(SUBSTRING(text FROM 1 FOR 12))) AS paper_id, "
            + "SUBSTRING(text FROM 13) AS paper_name");

    public static void main(String[] args) throws Exception {

        AlinkGlobalConfiguration.setPrintProcessInfo(true);

        c_2();

        c_3();

        c_4();

        c_5();

        c_6_1();
        c_6_2();

    }

    static void c_2() throws Exception {
        paper_author.lazyPrint(3, "< paper_author >");
        paper_conf.lazyPrint(3, "< paper_conf >");
        id_author.lazyPrint(3, "< id_author >");
        id_conf.lazyPrint(3, "< id_conf >");
        paper.lazyPrint(3, "< paper >");

        BatchOperator.execute();
    }

    static void c_3() throws Exception {
        BatchOperator<?> edges = new UnionBatchOp().linkFrom(
            paper_author.select("paper_id AS source_id, author_id AS target_id"),
            paper_conf.select("paper_id AS source_id, conf_id AS target_id")
        );

        edges
            .link(
                new DeepWalkBatchOp()
                    .setSourceCol("source_id")
                    .setTargetCol("target_id")
                    .setIsToUndigraph(true)
                    .setVectorSize(100)
                    .setWalkLength(10)
                    .setWalkNum(20)
                    .setNumIter(1)
            )
            .link(
                new AkSinkBatchOp()
                    .setFilePath(DATA_DIR + DEEPWALK_EMBEDDING)
                    .setOverwriteSink(true)
            );
        BatchOperator.execute();

        edges
            .link(
                new Node2VecBatchOp()
                    .setSourceCol("source_id")
                    .setTargetCol("target_id")
                    .setIsToUndigraph(true)
                    .setVectorSize(100)
                    .setWalkLength(10)
                    .setWalkNum(20)
                    .setP(2.0)
                    .setQ(0.5)
                    .setNumIter(1)
            )
            .link(
                new AkSinkBatchOp()
                    .setFilePath(DATA_DIR + NODE2VEC_EMBEDDING)
                    .setOverwriteSink(true)
            );
        BatchOperator.execute();

        BatchOperator id_type = new UnionBatchOp()
            .linkFrom(
                paper.select("paper_id AS node_id, 'P' AS node_type"),
                id_author.select("author_id AS node_id, 'A' AS node_type"),
                id_conf.select("conf_id AS node_id, 'C' AS node_type")
            );

        new MetaPath2VecBatchOp()
            .setMetaPath("APA,APCPA")
            .setVertexCol("node_id")
            .setTypeCol("node_type")
            .setSourceCol("source_id")
            .setTargetCol("target_id")
            .setIsToUndigraph(true)
            .setVectorSize(100)
            .setWalkLength(10)
            .setWalkNum(20)
            .setNumIter(1)
            .linkFrom(edges, id_type)
            .link(
                new AkSinkBatchOp()
                    .setFilePath(DATA_DIR + METAPATH2VEC_EMBEDDING)
                    .setOverwriteSink(true)
            );
        BatchOperator.execute();
    }

    static void c_4() throws Exception {
        for (String embedding_model_file :
            new String[]{DEEPWALK_EMBEDDING, NODE2VEC_EMBEDDING, METAPATH2VEC_EMBEDDING}
        ) {

            System.out.println("\n\n< " + embedding_model_file + " >\n");

            new AkSourceBatchOp()
                .setFilePath(DATA_DIR + embedding_model_file)
                .lazyPrint(3)
//                .lazyPrintStatistics()
                .link(
                    new VectorSummarizerBatchOp()
                        .setSelectedCol("vec")
                        .lazyPrintVectorSummary()
                );
            BatchOperator.execute();
        }
    }

    static void c_5() throws Exception {
        CsvSourceBatchOp author_name_label = new CsvSourceBatchOp()
            .setFilePath(DATA_DIR + "googlescholar.8area.author.label.txt")
            .setSchemaStr("author_labeled string, label int")
            .setFieldDelimiter(" ");

        BatchOperator author_id_label = new JoinBatchOp()
            .setJoinPredicate("author = author_labeled")
            .setSelectClause("author_id, label")
            .linkFrom(id_author, author_name_label);

        Utils.splitTrainTestIfNotExist(
            author_id_label,
            DATA_DIR + AUTHOR_LABEL_TRAIN,
            DATA_DIR + AUTHOR_LABEL_TEST,
            0.8
        );

        for (String embedding_model_file :
            new String[]{DEEPWALK_EMBEDDING, NODE2VEC_EMBEDDING, METAPATH2VEC_EMBEDDING}
        ) {

            System.out.println("\n\n< " + embedding_model_file + " >\n");

            classifyWithEmbedding(
                new AkSourceBatchOp().setFilePath(DATA_DIR + embedding_model_file)
            );

        }
    }

    static void classifyWithEmbedding(BatchOperator<?> graph_embedding) throws Exception {
        BatchOperator author_train = new JoinBatchOp()
            .setJoinPredicate("author_id = node")
            .setSelectClause("author_id, vec, label")
            .linkFrom(
                graph_embedding,
                new AkSourceBatchOp().setFilePath(DATA_DIR + AUTHOR_LABEL_TRAIN)
            );

        BatchOperator author_test = new LookupBatchOp()
            .setSelectedCols("author_id")
            .setOutputCols("vec")
            .linkFrom(
                graph_embedding,
                new AkSourceBatchOp().setFilePath(DATA_DIR + AUTHOR_LABEL_TEST)
            );

        new Softmax()
            .setVectorCol("vec")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .fit(author_train)
            .transform(author_test)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics("[ Using Softmax ]")
            );

        new KnnClassifier()
            .setVectorCol("vec")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .fit(author_train)
            .transform(author_test)
            .link(
                new EvalMultiClassBatchOp()
                    .setLabelCol("label")
                    .setPredictionCol("pred")
                    .lazyPrintMetrics("[ Using KnnClassifier ]")
            );

        BatchOperator.execute();
    }

    static void c_6_1() throws Exception {
        BatchOperator<?> edges = new UnionBatchOp().linkFrom(
            paper_author.select("paper_id AS source_id, author_id AS target_id"),
            paper_conf.select("paper_id AS source_id, conf_id AS target_id")
        );

        for (int walkNum : new int[]{10, 20, 50}) {
            edges
                .link(
                    new DeepWalkBatchOp()
                        .setSourceCol("source_id")
                        .setTargetCol("target_id")
                        .setIsToUndigraph(true)
                        .setVectorSize(100)
                        .setWalkLength(20)
                        .setWalkNum(walkNum)
                        .setNumIter(1)
                )
                .link(
                    new AkSinkBatchOp()
                        .setFilePath(DATA_DIR + String.valueOf(walkNum) + "_" + DEEPWALK_EMBEDDING)
                        .setOverwriteSink(true)
                );
            BatchOperator.execute();

            classifyWithEmbedding(
                new AkSourceBatchOp()
                    .setFilePath(DATA_DIR + String.valueOf(walkNum) + "_" + DEEPWALK_EMBEDDING)
            );
        }
    }

    static void c_6_2() throws Exception {
        BatchOperator<?> edges = new UnionBatchOp().linkFrom(
            paper_author.select("paper_id AS source_id, author_id AS target_id"),
            paper_conf.select("paper_id AS source_id, conf_id AS target_id"),
            new LookupBatchOp()
                .setSelectedCols("paper_id")
                .setOutputCols("target_id")
                .setMapKeyCols("paper_id")
                .setMapValueCols("conf_id")
                .linkFrom(paper_conf, paper_author)
                .select("author_id AS source_id, target_id")
        );

        for (int walkNum : new int[]{10, 20, 50}) {
            edges
                .link(
                    new DeepWalkBatchOp()
                        .setSourceCol("source_id")
                        .setTargetCol("target_id")
                        .setIsToUndigraph(true)
                        .setVectorSize(100)
                        .setWalkLength(20)
                        .setWalkNum(walkNum)
                        .setNumIter(1)
                )
                .link(
                    new AkSinkBatchOp()
                        .setFilePath(DATA_DIR + String.valueOf(walkNum) + "_" + DEEPWALK_EMBEDDING)
                        .setOverwriteSink(true)
                );
            BatchOperator.execute();

            classifyWithEmbedding(
                new AkSourceBatchOp()
                    .setFilePath(DATA_DIR + String.valueOf(walkNum) + "_" + DEEPWALK_EMBEDDING)
            );
        }
    }

}
