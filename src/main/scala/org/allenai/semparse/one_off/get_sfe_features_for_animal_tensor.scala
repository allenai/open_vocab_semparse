package org.allenai.semparse.one_off

import org.allenai.semparse.pipeline.base.SparkPmiComputer
import org.allenai.semparse.pipeline.base.TrainingDataFeatureComputer

import com.mattg.util.Dictionary
import com.mattg.util.ImmutableDictionary
import com.mattg.util.MutableConcurrentDictionary
import com.mattg.util.FileUtil

import org.json4s._
import org.json4s.JsonDSL._

import scala.collection.mutable

object get_sfe_features_for_animal_tensor {

  // Input files
  val sfeSpecFile = "/home/mattg/data/animal_tensor_kbc/sfe_spec.json"
  val nodePairFile = "/home/mattg/data/animal_tensor_kbc/node_pairs.tsv"
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
    val fileUtil = new FileUtil
    val pmiComputer = new SparkPmiComputer(pmiComputerParams, fileUtil)
    //pmiComputer.runPipeline()
    val entityDict = ImmutableDictionary.readFromFile(entityDictFile)
    val relationDict = ImmutableDictionary.readFromFile(relationDictFile)
    val relationFeatureDictionaries = new mutable.HashMap[String, Dictionary]
    val relationFeatures = fileUtil.mapLinesFromFile(pmiComputer.relWordFeatureFile, (line) => {
      val fields = line.split("\t")
      val relation = fields(0)
      val features = fields.drop(1)
      (relation -> features.toSet)
    }).toMap
    val writer = fileUtil.getFileWriter(finalFeatureFile)
    fileUtil.processFile(pmiComputer.filteredMidPairFeatureFile, (line: String) => {
      val fields = line.split("\t")
      val nodePair = fields(0).split(entitySeparator)
      val features = fields(1).split(featureSeparator)
      val sourceIndex = entityDict.getIndex(nodePair(0))
      val targetIndex = entityDict.getIndex(nodePair(1))
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
    })
    writer.close()
    fileUtil.mkdirs(featureDictionaryDir)
    for ((relation, dictionary) <- relationFeatureDictionaries) {
      dictionary.writeToFile(featureDictionaryDir + relation)
    }
  }
}
