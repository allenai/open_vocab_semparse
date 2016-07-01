package org.allenai.semparse.one_off

import org.allenai.semparse.pipeline.common.PreFilteredFeatureComputer
import org.allenai.semparse.pipeline.common.SparkPmiComputer
import org.allenai.semparse.pipeline.common.TrainingDataFeatureComputer

import com.mattg.util.Dictionary
import com.mattg.util.FileUtil
import com.mattg.util.ImmutableDictionary
import com.mattg.util.MutableConcurrentDictionary
import com.mattg.util.SpecFileReader

import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.data.NodePairSplit
import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.experiments.RelationMetadata
import edu.cmu.ml.rtw.pra.features.FeatureMatrix
import edu.cmu.ml.rtw.pra.features.NodePairSubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.models.LogisticRegressionModel
import edu.cmu.ml.rtw.pra.operations.HackyHanieOperation

import org.json4s._
import org.json4s.JsonDSL._

import java.io.FileWriter

import scala.collection.mutable
import scala.collection.JavaConverters._

object get_sfe_features_for_animal_tensor {
  implicit val formats = DefaultFormats
  val fileUtil = new FileUtil

  // Input files
  val sfeSpecFile = "/home/mattg/data/animal_tensor_kbc/sfe_spec.json"
  val nodePairFile = "/home/mattg/data/animal_tensor_kbc/all_node_pairs.tsv"
  val nodePairRelationFile = "/home/mattg/data/animal_tensor_kbc/node_pair_relations.tsv"
  val entityDictFile = "/home/mattg/data/animal_tensor_kbc/entity_dict.tsv"
  val relationDictFile = "/home/mattg/data/animal_tensor_kbc/relation_dict.tsv"

  // Output lives here
  val outdir = "/home/mattg/data/animal_tensor_kbc/sfe_features_pra/"
  val finalFeatureFile = s"${outdir}final_feature_tensor.tsv"
  val featureDictionaryDir = s"${outdir}feature_dictionaries/"

  val featureMatrixParams: JValue =
    ("sfe spec file" -> sfeSpecFile) ~
    ("mid pair file" -> nodePairFile) ~
    ("out dir" -> outdir) ~
    ("mid or mid pair" -> "mid pair")

  val pmiComputerParams: JValue =
    ("training data features" -> featureMatrixParams) ~
    ("features per word" -> 1000) ~
    ("mid or mid pair" -> "mid pair") ~
    ("mid pair word file" -> nodePairRelationFile)

  val entitySeparator = "__##__"
  val featureSeparator = " -#- "

  def main(args: Array[String]) {
    val pmiComputer = new SparkPmiComputer(pmiComputerParams, fileUtil)
    /*
    pmiComputer.runPipeline()
    convertFeatureFileToTensorFormat(
      pmiComputer.relWordFeatureFile,
      pmiComputer.filteredMidPairFeatureFile,
      finalFeatureFile,
      fileUtil
    )
    */
    getFeaturesForNodePairs(
      nodePairFile,
      sfeSpecFile,
      pmiComputer.relWordFeatureFile,
      finalFeatureFile
    )
    /*
    val splitDir = "/home/mattg/pra/splits/animals_with_negatives/"
    val relationToRun = "eat"
    val outFile = "eat_scores.tsv"
    testSfeWithSelectedFeatures(
      pmiComputer.relWordFeatureFile,
      sfeSpecFile,
      splitDir,
      relationToRun,
      outFile
    )
    */
  }

  def writeNodePairFeaturesToTensorFile(
    source: String,
    target: String,
    features: Seq[String],
    entityDict: Dictionary,
    relationDict: Dictionary,
    relationFeatures: Map[String, Set[String]],
    relationFeatureDictionaries: mutable.HashMap[String, Dictionary],
    writer: FileWriter
  ) {
    val sourceIndex = entityDict.getIndex(source)
    val targetIndex = entityDict.getIndex(target)
    for (relation <- relationFeatures.keys) {
      val relationIndex = relationDict.getIndex(relation)
      val featureDict = relationFeatureDictionaries.getOrElseUpdate(relation, new MutableConcurrentDictionary)
      for (feature <- features) {
        if (relationFeatures(relation).contains(feature)) {
          val featureIndex = featureDict.getIndex(feature) - 1
          writer.write(s"$sourceIndex\t$targetIndex\t$relationIndex\t$featureIndex\n")
        }
      }
    }
  }

  def getFeaturesForNodePairs(
    nodePairFile: String,
    specFile: String,
    relationFeatureFile: String,
    outputFile: String
  ) {
    val sfeParams = new SpecFileReader("/dev/null").readSpecFile(specFile)
    val writer = fileUtil.getFileWriter(outputFile)
    val outputterParams: JValue = ("outdir" -> "tmp_output/")
    val outputter = new Outputter(outputterParams, "/dev/null", "ignored", fileUtil)
    val relationMetadata = new RelationMetadata(sfeParams \ "relation metadata", "/dev/null", outputter, fileUtil)
    val graph = Graph.create(sfeParams \ "graph", "", outputter, fileUtil).get
    val featureGenerator = new NodePairSubgraphFeatureGenerator(
      sfeParams \ "node pair features",
      "ignored relation",
      relationMetadata,
      outputter,
      fileUtil = fileUtil
    )
    val entityDict = ImmutableDictionary.readFromFile(entityDictFile)
    val relationDict = ImmutableDictionary.readFromFile(relationDictFile)
    val relationFeatureDictionaries = new mutable.HashMap[String, Dictionary]
    val relationFeatures = loadRelationFeatureFiles(relationFeatureFile)
    val allAllowedFeatures = relationFeatures.flatMap(_._2).toSet
    val nodePairs = fileUtil.mapLinesFromFile(nodePairFile, (line) => {
      val fields = line.split(entitySeparator)
      (entityDict.getString(fields(0).toInt), entityDict.getString(fields(1).toInt))
    })
    var done = 0
    var chunkSize = 1024
    nodePairs.grouped(chunkSize).foreach(nodePairGroup => {
      val subgraphs = nodePairGroup.par.flatMap(nodePair => {
        val (source, target) = nodePair
        if (!graph.hasNode(source) || !graph.hasNode(target)) {
          Seq()
        } else {
          val instance = new NodePairInstance(graph.getNodeIndex(source), graph.getNodeIndex(target), true, graph)
          val subgraph = featureGenerator.getLocalSubgraph(instance)
          Seq((instance, subgraph))
        }
      })
      for ((instance, subgraph) <- subgraphs) {
        for (relation <- relationFeatures.keys) {
          val filteredSubgraph = featureGenerator.filterSubgraph(instance, subgraph, relation)
          val features = featureGenerator.extractFeaturesAsStrings(instance, filteredSubgraph)
          val sourceIndex = entityDict.getIndex(graph.getNodeName(instance.source))
          val targetIndex = entityDict.getIndex(graph.getNodeName(instance.target))
          val relationIndex = relationDict.getIndex(relation)
          val featureDict = relationFeatureDictionaries.getOrElseUpdate(relation, new MutableConcurrentDictionary)
          for (feature <- features) {
            if (relationFeatures(relation).contains(feature)) {
              val featureIndex = featureDict.getIndex(feature) - 1
              writer synchronized {
                writer.write(s"$sourceIndex\t$targetIndex\t$relationIndex\t$featureIndex\n")
              }
            }
          }
        }
      }
      done += chunkSize
      println(s"Done with $done")
    })
    writer.close()
    fileUtil.mkdirs(featureDictionaryDir)
    for ((relation, dictionary) <- relationFeatureDictionaries) {
      dictionary.writeToFile(featureDictionaryDir + relation)
    }
  }

  def loadRelationFeatureFiles(filename: String) = {
    fileUtil.mapLinesFromFile(filename, (line) => {
      val fields = line.split("\t")
      val relation = fields(0)
      val features = fields.drop(1)
      (relation -> features.toSet)
    }).toMap
  }

  def convertFeatureFileToTensorFormat(
    relationFeatureFile: String,
    nodePairFeatureFile: String,
    outputTensorFile: String
  ) {
    val entityDict = ImmutableDictionary.readFromFile(entityDictFile)
    val relationDict = ImmutableDictionary.readFromFile(relationDictFile)
    val relationFeatureDictionaries = new mutable.HashMap[String, Dictionary]
    val relationFeatures = loadRelationFeatureFiles(relationFeatureFile)
    val writer = fileUtil.getFileWriter(outputTensorFile)
    fileUtil.processFile(nodePairFeatureFile, (line: String) => {
      val fields = line.split("\t")
      val nodePair = fields(0).split(entitySeparator)
      val features = fields(1).split(featureSeparator)
      writeNodePairFeaturesToTensorFile(
        nodePair(0),
        nodePair(1),
        features,
        entityDict,
        relationDict,
        relationFeatures,
        relationFeatureDictionaries,
        writer
      )
    })
    writer.close()
    fileUtil.mkdirs(featureDictionaryDir)
    for ((relation, dictionary) <- relationFeatureDictionaries) {
      dictionary.writeToFile(featureDictionaryDir + relation)
    }
  }

  def testSfeWithSelectedFeatures(
    relationFeatureFile: String,
    specFile: String,
    splitDir: String,
    relationToRun: String,
    outfile: String
  ) {
    val relationFeatures = loadRelationFeatureFiles(relationFeatureFile)
    val allowedFeatures = relationFeatures(relationToRun)

    // SFE set up code
    val sfeParams = new SpecFileReader("/dev/null").readSpecFile(specFile)
    val outputterParams: JValue = ("outdir" -> "tmp_output/")
    val outputter = new Outputter(outputterParams, "/dev/null", "ignored", fileUtil)
    outputter.setRelation(relationToRun)
    val relationMetadata = new RelationMetadata(sfeParams \ "relation metadata", "/dev/null", outputter, fileUtil)
    val graph = Graph.create(sfeParams \ "graph", "", outputter, fileUtil).get
    val featureGenerator = new NodePairSubgraphFeatureGenerator(
      sfeParams \ "node pair features",
      relationToRun,
      relationMetadata,
      outputter,
      fileUtil = fileUtil
    )
    val splitParams = JString(splitDir)
    val split = new NodePairSplit(splitParams, "/dev/null", outputter, fileUtil)
    val modelParams = ("l1 weight" -> 0.005) ~ ("l2 weight" -> 1)
    val model = new LogisticRegressionModel[NodePairInstance](modelParams, outputter)
    val operationParams = ("features" -> (sfeParams \ "node pair features"))
    val operation = new HackyHanieOperation(
      operationParams,
      Some(graph),
      split,
      relationMetadata,
      outputter,
      fileUtil
    )

    println("Loading training and test data")
    val trainingData = split.getTrainingData(relationToRun, Some(graph))
    val testingData = split.getTestingData(relationToRun, Some(graph))

    println("Creating training feature matrix")
    val featureDict = new MutableConcurrentDictionary
    val matrixRows = trainingData.instances.par.map(instance => {
      val subgraph = featureGenerator.getLocalSubgraph(instance)
      val features = featureGenerator.extractFeaturesAsStrings(instance, subgraph)
      val keptFeatures = features.filter(allowedFeatures.contains).map(featureDict.getIndex)
      featureGenerator.createMatrixRow(instance, keptFeatures)
    }).seq

    println("Training the model")
    val featureNames = {
      val features = (1 until featureDict.size).par.map(i => {
          val name = featureDict.getString(i)
          if (name == null) {
            s"!!!!NULL FEATURE-${i}!!!!"
          } else {
            name
          }
      }).seq
      ("!!!!NULL FEATURE!!!!" +: features).toArray
    }
    model.train(new FeatureMatrix(matrixRows.asJava), trainingData, featureNames)

    println("Evaluating the model")
    operation.runRelationWithModel(relationToRun, model, featureDict)
  }
}
