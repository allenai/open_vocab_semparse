package org.allenai.semparse.pipeline.jklol

import org.scalatest._

import org.allenai.semparse.parse.Atom
import org.allenai.semparse.parse.Conjunction
import org.allenai.semparse.parse.Predicate

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

class LogicalFormToTrainingDataSpec extends FlatSpecLike with Matchers {
  val params: JValue = ("logical forms" -> ("data name" -> "tmp") ~ ("data directory" -> "tmp"))
  val fileUtil = new FileUtil
  val logicalFormToTrainingData = new LogicalFormToTrainingData(params, fileUtil)

  "logicalFormToTrainingData" should "produce correct output when the entities have commas" in {
    val logic = Some(Conjunction(Set(
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com, particularly blatant"))),
      Predicate("from", Seq(Atom("spam"), Atom("logic@imneverwrong.com,"))),
      Predicate("blatant", Seq(Atom("logic@imneverwrong.com,")))
    )))
    val sentence = "sentence"

    val json = "[[\"from\",\"spam\",\"logic@imneverwrong.com, particularly blatant\"], [\"from\",\"spam\",\"logic@imneverwrong.com,\"], [\"blatant\",\"logic@imneverwrong.com,\"]]"
    val expectedStrings = Seq(
      "\t\"spam\" \"logic@imneverwrong.com, particularly blatant\"\t\tfrom\t((word-rel \"from\") \"spam\" \"logic@imneverwrong.com, particularly blatant\")\t" + json + "\t" + sentence,
      "\t\"spam\" \"logic@imneverwrong.com,\"\t\tfrom\t((word-rel \"from\") \"spam\" \"logic@imneverwrong.com,\")\t" + json + "\t" + sentence,
      "\t\"logic@imneverwrong.com,\"\tblatant\t\t((word-cat \"blatant\") \"logic@imneverwrong.com,\")\t" + json + "\t" + sentence
    )

    val strings = logicalFormToTrainingData.logicalFormToTrainingData(sentence, logic)
    strings.size should be(expectedStrings.size)
    for (i <- 0 until expectedStrings.size) {
      strings(i) should be(expectedStrings(i))
    }
  }
}
