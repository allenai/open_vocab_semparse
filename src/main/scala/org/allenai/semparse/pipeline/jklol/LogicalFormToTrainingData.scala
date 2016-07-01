package org.allenai.semparse.pipeline.jklol

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.semparse.pipeline.science_data.SentenceToLogic
import org.allenai.semparse.parse._

class LogicalFormToTrainingData(
  params: JValue,
  fileUtil: FileUtil
) extends Step(None, fileUtil) {
  override val name = "Logical Form to Training Data"
  val validParams = Seq("logical forms")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceToLogic = new SentenceToLogic(params \ "logical forms", fileUtil)
  val outputFile = sentenceToLogic.outputFile.replace("logical_forms.txt", "training_data.tsv")

  override val inputs: Set[(String, Option[Step])] = Set((sentenceToLogic.outputFile, Some(sentenceToLogic)))
  override val outputs = Set(outputFile)
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    val outputLines = fileUtil.getLineIterator(sentenceToLogic.outputFile).flatMap(line => {
      val fields = line.split("\t")
      val sentence = fields(0)
      val logic = if (fields.length == 1 || fields(1).isEmpty) None else Some(Logic.fromString(fields(1)))
      logicalFormToTrainingData(sentence, logic)
    })
    fileUtil.writeLinesToFile(outputFile, outputLines.toSeq)
  }

  def logicalFormToTrainingData(sentence: String, logic: Option[Logic]): Seq[String] = {
    val fullJson = logic match {
      case None => "[]"
      case Some(logic) => logic.toJson
    }
    logic match {
      case Some(Conjunction(args)) => {
        args.map(_ match {
          case Predicate(predicate, arguments) => {
            // I'm trying to stay as close to the previous training data format as possible.  But,
            // the way the ids were formatted with Freebase data will not work for our noun
            // phrases.  So, I'm going to leave the id column blank as a single that the name
            // column should be used instead.
            val ids = ""
            val names = arguments.map("\"" + _ + "\"").mkString(" ")
            val catWord = if (arguments.size == 1) predicate else ""
            val relWord = if (arguments.size == 2) predicate else ""
            val wordRelOrCat = if (arguments.size == 1) "word-cat" else "word-rel"
            val lambdaExpression = s"""(($wordRelOrCat "${predicate}") $names)"""
            val fields = Seq(ids, names, catWord, relWord, lambdaExpression, fullJson, sentence)
            fields.mkString("\t")
          }
          case _ => throw new IllegalStateException
        }).toSeq
      }
      case _ => throw new IllegalStateException
    }
  }
}
