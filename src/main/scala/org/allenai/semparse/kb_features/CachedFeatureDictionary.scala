package org.allenai.semparse.kb_features

import com.jayantkrish.jklol.models.DiscreteVariable
import com.jayantkrish.jklol.tensor.SparseTensorBuilder
import com.jayantkrish.jklol.tensor.Tensor
import com.jayantkrish.jklol.tensor.TensorBuilder

import com.mattg.util.FileUtil

import collection.mutable
import collection.JavaConverters._
import collection.concurrent

abstract class CachedFeatureDictionary(
  featureFile: String,
  wordFeatureDictFile: String,
  saveCacheMisses: Boolean = true,
  fileUtil: FileUtil = new FileUtil
) {

  // Because the computation that populates these maps is idempotent, not _that_ expensive, and
  // very unlikely to have simultaneous requests for the same not-yet-computed key, we don't need
  // really sophisticated locking below; I'm just using a TrieMap to make sure some memory state
  // doesn't get corrupted by concurrent writes.
  lazy val features: concurrent.TrieMap[String, Seq[String]] = loadFeatures()
  val cacheMisses: concurrent.TrieMap[String, Seq[String]] = new concurrent.TrieMap
  val featureVectors: concurrent.TrieMap[String, mutable.Map[String, Tensor]] = new concurrent.TrieMap
  lazy val dictionaries: Map[String, DiscreteVariable] = loadDictionaries()

  val biasFeature = "bias"
  val unseenDictionary = new DiscreteVariable("unused", Set(biasFeature).asJava)

  if (saveCacheMisses) addShutdownHook()

  def addShutdownHook() {
    Runtime.getRuntime().addShutdownHook(new Thread() {
      override def run() {
        if (!cacheMisses.isEmpty) {
          println(s"Saving cache misses to $featureFile")
          val lines = cacheMisses.map(entry => {
            entry._1 + "\t" + entry._2.mkString(" -#- ")
          })
          fileUtil.writeLinesToFile(featureFile, lines, true)
        }
      }
    })
  }

  def getFeatureVector(key: String, word: String): Tensor = {
    val keyFeatureVectors = featureVectors.getOrElseUpdate(key, new mutable.HashMap)
    keyFeatureVectors.getOrElseUpdate(word, createVectorForWord(key, word))
  }

  def getNumCachedVectors() = features.size

  def getNumFeatures(word: String) = dictionaries.getOrElse(word, unseenDictionary).numValues()

  def isValidFeature(feature: String, word: String) =
    dictionaries.getOrElse(word, unseenDictionary).canTakeValue(feature)

  def getFeatureIndex(feature: String, word: String) =
    dictionaries.getOrElse(word, unseenDictionary).getValueIndex(feature)

  def loadFeatures() = {
    println(s"Loading CachedFeatureDictionary from file $featureFile")
    val f = new concurrent.TrieMap[String, Seq[String]]
    for (line <- fileUtil.getLineIterator(featureFile)) {
      val fields = line.split("\t")
      val key = fields(0)
      val featureArray = fields(1).split(" -#- ")
      f(key) = featureArray.toSeq
    }
    f
  }

  def loadDictionaries() = {
    println(s"Reading word features from $wordFeatureDictFile")
    fileUtil.getLineIterator(wordFeatureDictFile).map(line => {
      val fields = line.split("\t")
      val word = fields(0)
      val featureSet = fields.drop(1).toSet.asJava
      val dictionary = new DiscreteVariable("unused", featureSet)
      (word -> dictionary)
    }).toMap
  }

  def featureListToTensor(featureList: Seq[String], word: String): Tensor = {
    val builder = new SparseTensorBuilder(Array(0), Array(getNumFeatures(word)))
    for (feature <- featureList) {
      if (isValidFeature(feature, word)) {
        val featureIndex = getFeatureIndex(feature, word)
        builder.incrementEntry(1.0, featureIndex)
      }
    }
    builder.build()
  }

  def createVectorForWord(key: String, word: String): Tensor = {
    val keyFeatures = features.get(key) match {
      case Some(features) => features
      case None => {
        val tmp = computeFeaturesForKey(key)
        features(key) = tmp
        cacheMisses(key) = tmp
        tmp
      }
    }
    featureListToTensor(keyFeatures, word)
  }

  def computeFeaturesForKey(key: String): Seq[String]
}
