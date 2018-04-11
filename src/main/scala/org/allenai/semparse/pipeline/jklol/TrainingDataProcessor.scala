package org.allenai.semparse.pipeline.jklol

import collection.mutable

import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import com.mattg.pipeline.Step

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.parse

// I was was tired of having such a nasty pipeline of so many different scripts to run to get this
// working.  So, I moved Jayant's data processing script from python to scala.  I didn't change
// very much of the code, other than encapsulate the global state into these count objects seen at
// the top.  And now instead of running a python script four times with different options set in a
// bash script, you can run it all from scala code.  So, when this is actually ready for release, I
// will put another piece of scala code in a main driver that will check to see if the data needs
// to be processed, and call this code if it does.
//
// I checked the output of this code against Jayant's python code, and it matches exactly (except
// for one small sorting difference).  That's why there are some funny extra spaces in a few
// places.  Now that I know it matches exactly, I might change a bit of the formatting.

trait Query {
  def asString(): String
}

case class MidQuery(word: String) extends Query {
  override def asString() = s"($word)"
}
case class MidPairQuery(word: String, mid: String, midIsFirstArgument: Boolean) extends Query {
  override def asString() = s"($word, $midIsFirstArgument, $mid)"
}

// Keeps counts of cat(egory) and rel(ation) words.
class WordCounts {
  val catWordCounts: mutable.Map[String, Int] = new mutable.HashMap()
  val relWordCounts: mutable.Map[String, Int] = new mutable.HashMap()

  def observeCatWord(catWord: String) = catWordCounts.update(catWord, 1 + catWordCounts.getOrElse(catWord, 0))
  def observeRelWord(relWord: String) = relWordCounts.update(relWord, 1 + relWordCounts.getOrElse(relWord, 0))

  def getCatWords() = catWordCounts.keySet
  def getRelWords() = relWordCounts.keySet
  def getCatWordCount(catWord: String) = catWordCounts.getOrElse(catWord, 0)
  def getRelWordCount(relWord: String) = relWordCounts.getOrElse(relWord, 0)
}

// There are a number of things we need to count and keep track of as we go through the training
// file.  This does the counting, then passes the data structures off to an immutable object for
// when we're actually outputting stuff.
class TrainingDataBuilder {
  private val arg1Arg2Map: mutable.Map[String, mutable.Set[String]] = new mutable.HashMap()
  private val arg2Arg1Map: mutable.Map[String, mutable.Set[String]] = new mutable.HashMap()
  private val entityCounts: mutable.Map[String, Int] = new mutable.HashMap()
  private val entityPairCounts: mutable.Map[(String, String), Int] = new mutable.HashMap()
  private val logicalForms: mutable.ArrayBuffer[(String, JValue)] = new mutable.ArrayBuffer()
  private val relWordMidPairs: mutable.Map[String, mutable.Set[(String, String)]] = new mutable.HashMap()
  private val midPairWords: mutable.Map[(String, String), mutable.ArrayBuffer[String]] = new mutable.HashMap()
  private val catWordMids: mutable.Map[String, mutable.Set[String]] = new mutable.HashMap()
  private val midWords: mutable.Map[String, mutable.ArrayBuffer[String]] = new mutable.HashMap()
  private val queryMids: mutable.Map[Seq[Query], mutable.Set[String]] = new mutable.HashMap()

  def observeEntity(mid: String) { entityCounts.update(mid, 1 + entityCounts.getOrElse(mid, 0)) }
  def observeEntityPair(mid1: String, mid2: String) {
    val pair = (mid1, mid2)
    entityPairCounts.update(pair, 1 + entityPairCounts.getOrElse(pair, 0))
    arg1Arg2Map.getOrElseUpdate(mid1, new mutable.HashSet).add(mid2)
    arg2Arg1Map.getOrElseUpdate(mid2, new mutable.HashSet).add(mid1)
  }
  def observeQueryWithMid(query: Seq[Query], mid: String) {
    queryMids.getOrElseUpdate(query, new mutable.HashSet).add(mid)
  }
  def observeEntityWithWord(word: String, mid: String) {
    catWordMids.getOrElseUpdate(word, new mutable.HashSet).add(mid)
    midWords.getOrElseUpdate(mid, new mutable.ArrayBuffer) += word
  }
  def observeEntityPairWithWord(word: String, mid1: String, mid2: String) {
    relWordMidPairs.getOrElseUpdate(word, new mutable.HashSet).add((mid1, mid2))
    midPairWords.getOrElseUpdate((mid1, mid2), new mutable.ArrayBuffer) += word
  }
  def addLogicalForm(predicate: String, json: JValue) { logicalForms += Tuple2(predicate, json) }

  def build(): TrainingData = {
    new TrainingData(
      arg1Arg2Map.par.mapValues(_.toSet).seq.toMap,
      arg2Arg1Map.par.mapValues(_.toSet).seq.toMap,
      entityCounts.toMap,
      entityPairCounts.toMap,
      logicalForms.toSeq,
      relWordMidPairs.par.mapValues(_.toSet).seq.toMap,
      midPairWords.par.mapValues(_.toSeq).seq.toMap,
      catWordMids.par.mapValues(_.toSet).seq.toMap,
      midWords.par.mapValues(_.toSeq).seq.toMap,
      queryMids.par.mapValues(_.toSet).seq.toMap
    )
  }
}

class TrainingData(
  arg1Arg2Map: Map[String, Set[String]],
  arg2Arg1Map: Map[String, Set[String]],
  entityCounts: Map[String, Int],
  entityPairCounts: Map[(String, String), Int],
  logicalForms: Seq[(String, JValue)],
  relWordMidPairs: Map[String, Set[(String, String)]],
  midPairWords: Map[(String, String), Seq[String]],
  catWordMids: Map[String, Set[String]],
  midWords: Map[String, Seq[String]],
  queryMids: Map[Seq[Query], Set[String]]
) {
  def getEntities() = entityCounts.keySet
  def getMidWords(mid: String): Seq[String] = midWords.getOrElse(mid, Seq())
  def getEntityPairs() = entityPairCounts.keySet
  def getMidPairWords(midPair: (String, String)): Seq[String] = midPairWords.getOrElse(midPair, Seq())
  def getRelatedEntities(mid: String): Set[String] = {
    arg1Arg2Map.getOrElse(mid, Set()) ++ arg2Arg1Map.getOrElse(mid, Set())
  }
  def getEntityCount(mid: String) = entityCounts.getOrElse(mid, 0)
  def getEntityPairCount(mid1: String, mid2: String) = entityPairCounts.getOrElse((mid1, mid2), 0)
  def getArg1s() = arg1Arg2Map.keySet
  def getArg2sForArg1(mid1: String): Set[String] = arg1Arg2Map.getOrElse(mid1, Set())
  def getArg1sForArg2(mid2: String): Set[String] = arg2Arg1Map.getOrElse(mid2, Set())
  def getCatWordMids(word: String): Set[String] = catWordMids.getOrElse(word, Set())
  def getRelWordMidPairs(word: String): Set[(String, String)] = relWordMidPairs.getOrElse(word, Set())
  def getQueries() = queryMids.keySet
  def getMidsForQuery(query: Seq[Query]) = queryMids.getOrElse(query, Set())
  def getLogicalForms() = logicalForms
}

class TrainingDataProcessor(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats

  val validParams = Seq(
    "training data file",
    "training data creator",
    "data name",
    "word count threshold",
    "lines to use"
  )
  JsonHelper.ensureNoExtras(params, "trainer", validParams)

  val trainingDataCreator: Option[Step] = (params \ "training data creator") match {
    case JNothing => None
    case jval => {
      (jval \ "type") match {
        case JString("science sentence processor") => {
          // TODO(matt): this is likely a little bit broken, but I'm not using this anymore.
          //Some(new LogicalFormToTrainingData(jval.removeField(_._1 == "type"), fileUtil))
          throw new IllegalStateException("This was removed")
        }
        case _ => throw new IllegalStateException("unrecognized training data creator")
      }
    }
  }
  val trainingDataFile = trainingDataCreator match {
    case None => (params \ "training data file").extract[String]
    case Some(creator) => creator.outputs.head
  }

  val dataName = (params \ "data name").extract[String]
  val wordCountThreshold = (params \ "word count threshold").extract[Int]
  val linesToUse: Option[Int] = (params \ "lines to use") match {
    case JNothing => None
    case jval => Some(jval.extract[Int])
  }

  val outDir = s"data/$dataName"
  val wordFile = s"$outDir/words.lisp"
  val entityFile = s"$outDir/entities.lisp"
  val jointEntityFile = s"$outDir/joint_entities.lisp"
  val predicateRankingLfFile = s"$outDir/predicate_ranking_lf.lisp"
  val queryRankingLfFile = s"$outDir/query_ranking_lf.lisp"
  val midFile = s"$outDir/training_mids.tsv"
  val midPairFile = s"$outDir/training_mid_pairs.tsv"
  val midWordFile = s"$outDir/training_mid_words.tsv"
  val midPairWordFile = s"$outDir/training_mid_pair_words.tsv"

  override val paramFile = s"$outDir/tdp_params.json"
  override val inProgressFile = s"$outDir/tdp_in_progress"
  override val name = "Training data processor"
  override def inputs = Set((trainingDataFile, trainingDataCreator))
  override val outputs = Set(
    wordFile,
    entityFile,
    jointEntityFile,
    predicateRankingLfFile,
    queryRankingLfFile,
    midFile,
    midPairFile,
    midWordFile,
    midPairWordFile
  )

  val entityRegex = "\"([^\"]*)\"".r
  val catWordRegex = "word-cat \"([^\"]*)\"".r
  val relWordRegex = "word-rel \"([^\"]*)\"".r
  fileUtil.mkdirs(outDir)

  override def _runStep() {
    // TODO(matt): this is slow on large files, as all processing is sequential.  Is there some way
    // to parallelize this?
    println("Doing initial pass over the training data to get word counts")
    val wordCounts = getWordCountsFromTrainingFile()
    println("Reading the training data into memory")
    val trainingData = readTrainingData(wordCounts)

    // These five lisp files are all used as input to various parts of the semantic parsing code.
    println("Outputting word.lisp")
    outputWordFile(trainingData, wordCounts)
    println("Outputting entities.lisp")
    outputEntityFile(trainingData)
    println("Outputting joint_entities.lisp")
    val queryIndex = outputJointEntityFile(trainingData)
    println("Outputting predicate_ranking_lf.lisp")
    outputPredicateRankingLogicalForms(trainingData)
    println("Outputting query_ranking_lf.lisp")
    outputQueryRankingLogicalForms(trainingData, queryIndex)

    // And these tsv files are used by the PMI pipeline, and to precompute features for MIDs and
    // MID pairs in the training data.
    println("Outputting training-mid-words.tsv")
    outputMidWordsFile(trainingData)
    println("Outputting training-mid-pair-words.tsv")
    outputMidPairWordsFile(trainingData)
    println("Outputting training-mids.tsv")
    outputMidList(trainingData)
    println("Outputting training-mid-pairs.tsv")
    outputMidPairList(trainingData)
  }

  def jsonToQuerySeq(json: JValue, mid: String): Seq[Query] = {
    val queries = json.extract[Seq[Seq[String]]]
    queries.flatMap(query => {
      if (query.size == 2) {
        if (query(1) == mid) {
          Seq(MidQuery(query(0)))
        } else {
          Seq()
        }
      } else {
        if (query(1) == mid) {
          Seq(MidPairQuery(query(0), query(2), false))
        } else if (query(2) == mid) {
          Seq(MidPairQuery(query(0), query(1), true))
        } else {
          Seq()
        }
      }
    }).sortBy(_.asString)
  }

  def getWordsFromField(field: String): Seq[String] = {
    if (field.isEmpty) {
      Seq()
    } else {
      field.split(" ").map(_.replace("\\", "\\\\"))
    }
  }

  def getTrainingFileIterator() = linesToUse match {
    case None => fileUtil.getLineIterator(trainingDataFile)
    case Some(lines) => fileUtil.getLineIterator(trainingDataFile).take(lines)
  }

  // We need to get cat and rel word counts in a pass over the training file, so we can use the
  // counts to do some thresholding in the second pass through the file.
  def getWordCountsFromTrainingFile(): WordCounts = {
    val wordCounts = new WordCounts
    for (line <- getTrainingFileIterator) {
      val fields = line.split("\t")
      val catWordField = fields(2).trim()
      val relWordField = fields(3).trim()

      if (!catWordField.isEmpty) {
        val catWords = getWordsFromField(catWordField)
        for (catWord <- catWords) {
          wordCounts.observeCatWord(catWord)
        }
      }

      if (!relWordField.isEmpty) {
        val relWords = getWordsFromField(relWordField)
        for (relWord <- relWords) {
          wordCounts.observeRelWord(relWord)
        }
      }
    }
    wordCounts
  }

  def readTrainingData(wordCounts: WordCounts): TrainingData = {
    val trainingData = new TrainingDataBuilder
    for (line <- getTrainingFileIterator) {
      val fields = line.split("\t")
      val mids = if (fields(0).isEmpty) {
        val names = fields(1).split("\" \"")
        if (names.length == 1) {
          val mid = names(0).drop(1).dropRight(1)
          Array(mid)
        } else {
          val newNames = new Array[String](2)
          newNames(0) = names(0).drop(1)
          newNames(1) = names(1).dropRight(1)
          newNames
        }
      } else {
        fields(0).split(" ")
      }
      val catWords = getWordsFromField(fields(2))
      val relWords = getWordsFromField(fields(3))
      val counts = catWords.map(w => wordCounts.getCatWordCount(w)) ++ relWords.map(w => wordCounts.getRelWordCount(w))
      val missedThreshold = counts.filter(_ <= wordCountThreshold)
      if (missedThreshold.size == 0) {
        val logicalForm = fields(4)
        val json = parse(fields(5).replace("\\", "\\\\"))
        trainingData.addLogicalForm(logicalForm, json)
        for (mid <- mids) {
          trainingData.observeEntity(mid)
          val querySeq = jsonToQuerySeq(json, mid)
          trainingData.observeQueryWithMid(querySeq, mid)
        }
        for (catWord <- catWords) {
          trainingData.observeEntityWithWord(catWord, mids(0))
        }
        for (relWord <- relWords) {
          trainingData.observeEntityPairWithWord(relWord, mids(0), mids(1))
        }
        if (mids.length == 2) {
          val arg1 = mids(0)
          val arg2 = mids(1)
          trainingData.observeEntityPair(arg1, arg2)
        }
      }
    }
    trainingData.build()
  }

  def outputEntityFile(trainingData: TrainingData) {
    val writer = fileUtil.getFileWriter(entityFile)
    val entities = trainingData.getEntities.toSeq.sorted
    writer.write("(define entity-histogram (make-histogram \n")

    for (entity <- entities) {
      writer.write("(list \"%s\" %d)\n".format(entity, trainingData.getEntityCount(entity)))
    }
    writer.write("))\n")
    writer.write("(define entities (histogram-to-dictionary entity-histogram))\n")

    writer.write("(define entity-tuple-histogram (make-histogram \n")
    for (arg1 <- trainingData.getArg1s.toSeq.sorted) {
      for (arg2 <- trainingData.getArg2sForArg1(arg1).toSeq.sorted) {
        val count = trainingData.getEntityPairCount(arg1, arg2)
        writer.write("(list (list \"%s\" \"%s\") %d)\n".format(arg1, arg2, count))
      }
    }
    writer.write("))\n")
    writer.write("(define entity-tuples (histogram-to-dictionary entity-tuple-histogram))\n")

    writer.write("(define related-entities (array \n")
    for (entity <- entities) {
      val relatedEntities = trainingData.getRelatedEntities(entity).toSeq.sorted.map(e => "\"" + e + "\"").mkString(" ")
        writer.write(s"(array $relatedEntities )\n")
    }
    writer.write("))\n")

    writer.write("(define arg1-arg2-map (array \n")
    for (entity <- entities) {
      val arg1s = trainingData.getArg2sForArg1(entity).toSeq.sorted.map(e => "\"" + e + "\"").mkString(" ")
      writer.write(s"(make-dset entities (array $arg1s ))\n")
    }
    writer.write("))\n")

    writer.write("(define arg2-arg1-map (array \n")
    for (entity <- entities) {
      val arg2s = trainingData.getArg1sForArg2(entity).toSeq.sorted.map(e => "\"" + e + "\"").mkString(" ")
      writer.write(s"(make-dset entities (array $arg2s ))\n")
    }
    writer.write("))\n")
    writer.close()
  }

  def outputWordFile(trainingData: TrainingData, wordCounts: WordCounts) {
    val writer = fileUtil.getFileWriter(wordFile)
    val catWords = (Seq("<UNK>") ++
      wordCounts.getCatWords.filter(w => wordCounts.getCatWordCount(w) > wordCountThreshold).toSeq.sorted)
    val relWords = (Seq("<UNK>") ++
      wordCounts.getRelWords.filter(w => wordCounts.getRelWordCount(w) > wordCountThreshold).toSeq.sorted)
    writer.write("(define cat-words (make-dictionary \n")
    for (catWord <- catWords) {
        writer.write("\"" + catWord + "\"\n")
    }
    writer.write("))\n")

    writer.write("(define rel-words (make-dictionary \n")
    for (relWord <- relWords) {
        writer.write("\"" + relWord + "\"\n")
    }
    writer.write("))\n")

    writer.write("(define cat-word-entities (array \n")
    for (catWord <- catWords) {
      val mids = if (catWord == "<UNK>") {
        catWords.filter(w => wordCounts.getCatWordCount(w) <= wordCountThreshold).flatMap(w => {
          trainingData.getCatWordMids(w)
        })
      } else {
        trainingData.getCatWordMids(catWord)
      }
      val midStr = mids.toSeq.sorted.map(e => "\"" + e + "\"").mkString(" ")
      writer.write(s"(make-dset entities (array $midStr))\n")
    }
    writer.write("))\n")

    writer.write("(define rel-word-entities (array \n")
    for (relWord <- relWords) {
      val midPairs = if (relWord == "<UNK>") {
        relWords.filter(w => wordCounts.getRelWordCount(w) <= wordCountThreshold).flatMap(w => {
          trainingData.getRelWordMidPairs(w)
        })
      } else {
        trainingData.getRelWordMidPairs(relWord)
      }
      val midStr = midPairs.toSeq.sorted.map(e => "(list \"" + e._1 + "\" \"" + e._2 + "\")").mkString(" ")
      writer.write(s"(make-dset entity-tuples (array $midStr))\n")
    }
    writer.write("))\n")
    writer.close()
  }

  def outputJointEntityFile(trainingData: TrainingData): Map[Seq[Query], Int] = {
    val queryIndex = new mutable.HashMap[Seq[Query], Int]
    val writer = fileUtil.getFileWriter(jointEntityFile)

    val queries = trainingData.getQueries()
    writer.write("(define joint-entities (array \n")
    for ((query, index) <- queries.toSeq.sortBy(_.map(_.asString).mkString("\t")).zipWithIndex) {
      queryIndex(query) = index
      val mids = trainingData.getMidsForQuery(query).toSeq.sorted.map(e => "\"" + e + "\"").mkString(" ")
      writer.write(s"(make-dset entities (array $mids))\n")
    }
    writer.write("))\n")
    writer.close()

    queryIndex.toMap
  }

  def replaceLast(string: String, toReplace: String, replaceWith: String): String = {
    val lastIndex = string.lastIndexOf(toReplace)
    if (lastIndex < 0) {
      string
    } else {
      val tail = string.substring(lastIndex).replace(toReplace, replaceWith)
      string.substring(0, lastIndex) + tail
    }
  }

  def outputPredicateRankingLogicalForms(trainingData: TrainingData) {
    val writer = fileUtil.getFileWriter(predicateRankingLfFile)
    val entities = trainingData.getEntities

    writer.write("(define training-inputs (array \n")
    for ((logicalForm, json) <- trainingData.getLogicalForms) {
      val entityPart = logicalForm.substring(logicalForm.indexOf(")"))
      val entityNames = entityRegex.findAllIn(entityPart).toSeq
      val entityNameSet = entityNames.toSet
      // If this is a rel word, and both arguments are the same entity, skip it.
      if (entityNames.size == entityNameSet.size) {
        val varNames = new mutable.ArrayBuffer[String]
        var subbedLogicalForm = logicalForm
        for ((mid, index) <- entityNames.zipWithIndex) {
          val varName = s"var$index"
          val negVarName = s"neg-var$index"
          varNames += varName
          varNames += negVarName
          subbedLogicalForm = replaceLast(subbedLogicalForm, mid, s"$varName $negVarName")
        }

        val word = if (entityNames.size == 1) {
          catWordRegex.findFirstMatchIn(subbedLogicalForm).get.group(1)
        } else {
          relWordRegex.findFirstMatchIn(subbedLogicalForm).get.group(1)
        }
        writer.write("(list (quote (lambda ( ")
        writer.write(varNames.mkString(" "))
        writer.write(" ) ")
        writer.write(subbedLogicalForm)
        writer.write(" )) (list ")
        writer.write(entityNames.mkString(" "))
        writer.write(" ) ")
        writer.write("\"" + word + "\" )\n")
      }
    }
    writer.write("))\n")
    writer.close()
  }

  def outputQueryRankingLogicalForms(trainingData: TrainingData, queryIndex: Map[Seq[Query], Int]) {
    val writer = fileUtil.getFileWriter(queryRankingLfFile)
    val entities = trainingData.getEntities

    writer.write("(define training-inputs (array \n")
    for ((logicalForm, json) <- trainingData.getLogicalForms) {
      val predicates = json.extract[Seq[Seq[String]]]
      val relations = predicates.filter(s => s.size == 3 && entities.contains(s(1)) && entities.contains(s(2)))
      val midPairs = relations.map(s => ("\"" + s(1) + "\"", "\"" + s(2) + "\""))
      val arg1s = midPairs.groupBy(_._1).mapValues(_.map(_._2))
      val arg2s = midPairs.groupBy(_._2).mapValues(_.map(_._1))
      val entityPart = logicalForm.substring(logicalForm.indexOf(")"))
      val entityNames = entityRegex.findAllIn(entityPart).toSeq
      if (entityNames.size > 2) {
        System.err.println(s"Bad logical form: $logicalForm, $json")
      } else {
        val entityNameSet = entityNames.toSet
        // If this is a rel word, and both arguments are the same entity, skip it.
        if (entityNames.size == entityNameSet.size) {
          for (entityName <- entityNames.sorted) {
            val varName = "var"
            val negVarName = "neg-var"
            var subbedLogicalForm = replaceLast(logicalForm, entityName, s"$varName $negVarName")

            for (otherEntity <- (entityNameSet - entityName)) {
              subbedLogicalForm = replaceLast(subbedLogicalForm, otherEntity, s"$otherEntity $otherEntity")
            }

            val arg1Str = "(list " + arg2s.getOrElse(entityName, Set()).toSeq.sorted.mkString(" ") + ")"
            val arg2Str = "(list " + arg1s.getOrElse(entityName, Set()).toSeq.sorted.mkString(" ") + ")"

            val query = jsonToQuerySeq(json, entityName.replace("\"", ""))
            val index = queryIndex(query)

            writer.write("(list (quote (lambda (var neg-var) ")
            writer.write(subbedLogicalForm)
            writer.write(" )) (list ")
            writer.write(entityName)
            writer.write(" ) ")
            writer.write(arg1Str + "   " + arg2Str + "   " + index + " )\n")
          }
        }
      }
    }
    writer.write("))\n")
    writer.close()
  }

  def outputMidList(trainingData: TrainingData) {
    val mids = trainingData.getEntities().toSeq.sorted
    fileUtil.writeLinesToFile(midFile, mids)
  }

  def outputMidPairList(trainingData: TrainingData) {
    val midPairs = trainingData.getEntityPairs().toSeq.sorted.map(x => x._1 + "__##__" + x._2)
    fileUtil.writeLinesToFile(midPairFile, midPairs)
  }

  def outputMidWordsFile(trainingData: TrainingData) {
    val mids = trainingData.getEntities().toSeq.sorted
    val midWords = mids.map(m => (m, trainingData.getMidWords(m))).map(entry => {
      entry._1 + "\t" + entry._2.mkString("\t")
    })
    fileUtil.writeLinesToFile(midWordFile, midWords)
  }

  def outputMidPairWordsFile(trainingData: TrainingData) {
    val midPairs = trainingData.getEntityPairs().toSeq.sorted
    val midPairWords = midPairs.map(m => (m, trainingData.getMidPairWords(m))).map(entry => {
      entry._1._1 + "__##__" + entry._1._2 + "\t" + entry._2.mkString("\t")
    })
    fileUtil.writeLinesToFile(midPairWordFile, midPairWords)
  }
}
