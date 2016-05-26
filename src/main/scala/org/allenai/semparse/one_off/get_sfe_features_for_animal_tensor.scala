package org.allenai.semparse.one_off

import org.allenai.semparse.pipeline.base.PreFilteredFeatureComputer
import org.allenai.semparse.pipeline.base.SparkPmiComputer
import org.allenai.semparse.pipeline.base.TrainingDataFeatureComputer

import com.mattg.util.Dictionary
import com.mattg.util.FileUtil
import com.mattg.util.ImmutableDictionary
import com.mattg.util.MutableConcurrentDictionary
import com.mattg.util.SpecFileReader

import org.json4s._
import org.json4s.JsonDSL._

import java.io.FileWriter

import scala.collection.mutable

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
    val graphDirectory = (sfeParams \ "graph").extract[String]
    val computer = new PreFilteredFeatureComputer(sfeParams, fileUtil)
    val writer = fileUtil.getFileWriter(outputFile)
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
      val nodePairFeatures = nodePairGroup.par.map(nodePair => {
        val (source, target) = nodePair
        val features = computer.computeMidPairFeatures(source, target).filter(f => allAllowedFeatures.contains(f))
        (source, target, features)
      }).seq
      for ((source, target, features) <- nodePairFeatures) {
        writeNodePairFeaturesToTensorFile(
          source,
          target,
          features,
          entityDict,
          relationDict,
          relationFeatures,
          relationFeatureDictionaries,
          writer
        )
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
}
