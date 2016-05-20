package org.allenai.semparse.one_off

import com.mattg.util.FileUtil

import org.allenai.semparse.parse.Atom
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.pipeline.science_data.Helper

object debug_format_to_training_data {
  val inputFile = "/home/mattg/clone/semparse/data/science_sentences.txt"
  val outputFile = "/home/mattg/clone/semparse/data/science_sentences_training_data.txt"
  val fileUtil = new FileUtil
  def NOT_main(args: Array[String]) {
    val lines = fileUtil.readLinesFromFile(inputFile).par
    val parsed = lines.map(parseDebugLine)
    val outputLines = parsed.flatMap(Helper.logicalFormToTrainingData)
    fileUtil.writeLinesToFile(outputFile, outputLines.seq)
  }

  def parseDebugLine(line: String): (String, Set[Predicate]) = {
    val fields = line.split(" -> ")
    val sentence = fields(0)
    if (fields.length == 1) return (sentence, Set())
    val lf = fields(1)
    val lf_fields = lf.split("\\) ")
    val predicates = lf_fields.map(lf_field => {
      val paren = lf_field.indexOf("(")
      val predicate = lf_field.substring(0, paren)
      val args = lf_field.substring(paren + 1).split(", ").map(arg => {
        if (arg.endsWith(")")) {
          Atom(arg.substring(0, arg.length - 1))
        } else {
          Atom(arg)
        }
      })
      Predicate(predicate, args.toSeq)
    }).toSet
    (sentence, predicates)
  }
}
