package org.allenai.semparse.pipeline.jklol

import com.mattg.util.FakeFileUtil

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._

class TrainingDataProcessorSpec extends FlatSpecLike with Matchers {
  val clueWebDataFilename = "/clueweb_data_file"
  val clueWebDataFileContents =
    """/m/01cb_k /m/03z3fl	"H. pylori" "J. Robin Warren"		discovered_by	((word-rel "discovered_by") "/m/01cb_k" "/m/03z3fl")	[["discovered_by","/m/01cb_k","/m/03z3fl"], ["discovered_by","/m/01cb_k","/m/023fbk"]]	In 1982 , H. pylori was discovered by Barry J. Marshall and J. Robin Warren .
/m/052mx	"Mercedes Benz"	450		((word-cat "450") "/m/052mx")	[["450","/m/052mx"], ["town","/m/052mx"], ["lincoln","/m/052mx"], ["cars","/m/052mx"], ["new","/m/052mx"], ["available","/m/0ctrp"]]	With Dial 7 you are provided access to a fleet of over 450 new Lincoln Town Cars , Mercedes Benz and Cadillac Luxury Sedans as well as the most luxurious selection of stretch limousines and SUVs available .
/m/052mx	"Mercedes Benz"	town		((word-cat "town") "/m/052mx")	[["450","/m/052mx"], ["town","/m/052mx"], ["lincoln","/m/052mx"], ["cars","/m/052mx"], ["new","/m/052mx"], ["available","/m/0ctrp"]]	With Dial 7 you are provided access to a fleet of over 450 new Lincoln Town Cars , Mercedes Benz and Cadillac Luxury Sedans as well as the most luxurious selection of stretch limousines and SUVs available .
/m/0478h /m/01dvt	"Ju 87" "B-2"		N/N	((word-rel "N/N") "/m/0478h" "/m/01dvt")	[["N/N","/m/0478h","/m/01dvt"], ["luftwaffe","/m/02gyr7"], ["oberst","/m/02gyr7"]]	The paint scheme is identical to that of the Ju 87 B-2 flown by Luftwaffe Oberst Hans Ulrich Rudel in WW II .
"""

  val expectedClueWebPredicateLfOutput =
    """(define training-inputs (array 
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "discovered_by") var0 neg-var0 var1 neg-var1) )) (list "/m/01cb_k" "/m/03z3fl" ) "discovered_by" )
(list (quote (lambda ( var0 neg-var0 ) ((word-cat "450") var0 neg-var0) )) (list "/m/052mx" ) "450" )
(list (quote (lambda ( var0 neg-var0 ) ((word-cat "town") var0 neg-var0) )) (list "/m/052mx" ) "town" )
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "N/N") var0 neg-var0 var1 neg-var1) )) (list "/m/0478h" "/m/01dvt" ) "N/N" )
))
"""

  val expectedClueWebQueryLfOutput =
    """(define training-inputs (array 
(list (quote (lambda (var neg-var) ((word-rel "discovered_by") var neg-var "/m/03z3fl" "/m/03z3fl") )) (list "/m/01cb_k" ) (list )   (list "/m/03z3fl")   3 )
(list (quote (lambda (var neg-var) ((word-rel "discovered_by") "/m/01cb_k" "/m/01cb_k" var neg-var) )) (list "/m/03z3fl" ) (list "/m/01cb_k")   (list )   4 )
(list (quote (lambda (var neg-var) ((word-cat "450") var neg-var) )) (list "/m/052mx" ) (list )   (list )   0 )
(list (quote (lambda (var neg-var) ((word-cat "town") var neg-var) )) (list "/m/052mx" ) (list )   (list )   0 )
(list (quote (lambda (var neg-var) ((word-rel "N/N") "/m/0478h" "/m/0478h" var neg-var) )) (list "/m/01dvt" ) (list "/m/0478h")   (list )   2 )
(list (quote (lambda (var neg-var) ((word-rel "N/N") var neg-var "/m/01dvt" "/m/01dvt") )) (list "/m/0478h" ) (list )   (list "/m/01dvt")   1 )
))
"""

  val scienceDataFilename = "/science_data_file"
  val scienceDataFileContents =
"""	"spring" "wildflower"		beautiful_with	((word-rel "beautiful_with") "spring" "wildflower")	[["beautiful_with","spring","wildflower in bloom"], ["beautiful_with","spring","wildflower"], ["in","wildflower","bloom"]]	Spring is especially beautiful with the wildflowers in bloom.
	"ovum" "yolk"		contain	((word-rel "contain") "ovum" "yolk")	[["be","spherical","ovum"], ["contain","ovum","little yolk"], ["contain","ovum","yolk"], ["little","yolk"]]	The ovum is spherical and contains little yolk.
	"strategy"	remediation		((word-cat "remediation") "strategy")	[["&#039;","they","remediation strategy"], ["&#039;_obj","strategy","ve"], ["&#039;_obj","remediation strategy","ve"], ["&#039;","they","ve take"], ["&#039;_obj","remediation strategy","ve take"], ["&#039;_obj","strategy","ve take"], ["&#039;","they","strategy"], ["remediation","strategy"], ["&#039;_Csubj_commit","they","re"], ["&#039;","they","ve"]]	whatever remediation strategy they&#039;ve taken, they&#039;re committed to.
	"galaxy cluster" "cosmology"		lab_for	((word-rel "lab_for") "galaxy cluster" "cosmology")	[["lab_for","galaxy cluster","cosmology"], ["lab_for","cluster","cosmology"], ["galaxy","cluster"]]	Galaxy clusters are great labs for cosmology.
	"mold" "form"		form	((word-rel "form") "mold" "form")	[["form","mold","form"], ["form","latex mold","form"], ["form","latex mold be peel away","clay form"], ["latex","mold"], ["form","latex mold be peel away","form"], ["clay","form"], ["form","mold","clay form"], ["form","latex mold","clay form"]]	The latex mold is peeled away form the clay form.
"""

  val expectedSciencePredicateLfOutput =
    """(define training-inputs (array 
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "beautiful_with") var0 neg-var0 var1 neg-var1) )) (list "spring" "wildflower" ) "beautiful_with" )
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "contain") var0 neg-var0 var1 neg-var1) )) (list "ovum" "yolk" ) "contain" )
(list (quote (lambda ( var0 neg-var0 ) ((word-cat "remediation") var0 neg-var0) )) (list "strategy" ) "remediation" )
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "lab_for") var0 neg-var0 var1 neg-var1) )) (list "galaxy cluster" "cosmology" ) "lab_for" )
(list (quote (lambda ( var0 neg-var0 var1 neg-var1 ) ((word-rel "form") var0 neg-var0 var1 neg-var1) )) (list "mold" "form" ) "form" )
))
"""

  val expectedScienceQueryLfOutput =
    """(define training-inputs (array 
(list (quote (lambda (var neg-var) ((word-rel "beautiful_with") var neg-var "wildflower" "wildflower") )) (list "spring" ) (list )   (list "wildflower")   2 )
(list (quote (lambda (var neg-var) ((word-rel "beautiful_with") "spring" "spring" var neg-var) )) (list "wildflower" ) (list "spring")   (list )   3 )
(list (quote (lambda (var neg-var) ((word-rel "contain") var neg-var "yolk" "yolk") )) (list "ovum" ) (list )   (list "yolk")   1 )
(list (quote (lambda (var neg-var) ((word-rel "contain") "ovum" "ovum" var neg-var) )) (list "yolk" ) (list "ovum")   (list )   5 )
(list (quote (lambda (var neg-var) ((word-cat "remediation") var neg-var) )) (list "strategy" ) (list )   (list )   0 )
(list (quote (lambda (var neg-var) ((word-rel "lab_for") "galaxy cluster" "galaxy cluster" var neg-var) )) (list "cosmology" ) (list "galaxy cluster")   (list )   8 )
(list (quote (lambda (var neg-var) ((word-rel "lab_for") var neg-var "cosmology" "cosmology") )) (list "galaxy cluster" ) (list )   (list "cosmology")   7 )
(list (quote (lambda (var neg-var) ((word-rel "form") "mold" "mold" var neg-var) )) (list "form" ) (list "mold")   (list )   4 )
(list (quote (lambda (var neg-var) ((word-rel "form") var neg-var "form" "form") )) (list "mold" ) (list )   (list "form")   6 )
))
"""

  def getFileUtil() = {
    val f = new FakeFileUtil
    f.addFileToBeRead(clueWebDataFilename, clueWebDataFileContents)
    f.addFileToBeRead(scienceDataFilename, scienceDataFileContents)
    f
  }

  val baseParams: JValue = ("data name" -> "testing") ~ ("word count threshold" -> 0)

  "outputPredicateRankingLogicalForms" should "give the correct output on clueweb input" in {
    val extraParams: JValue = ("training data file" -> clueWebDataFilename)
    val params = baseParams merge extraParams
    val fileUtil = getFileUtil()
    val processor = new TrainingDataProcessor(params, fileUtil)
    fileUtil.addExpectedFileWritten(processor.predicateRankingLfFile, expectedClueWebPredicateLfOutput)
    fileUtil.onlyAllowExpectedFiles()
    val wordCounts = processor.getWordCountsFromTrainingFile()
    val trainingData = processor.readTrainingData(wordCounts)
    processor.outputPredicateRankingLogicalForms(trainingData)
    fileUtil.expectFilesWritten()
  }

  it should "give correct output on science input" in {
    val extraParams: JValue = ("training data file" -> scienceDataFilename)
    val params = baseParams merge extraParams
    val fileUtil = getFileUtil()
    val processor = new TrainingDataProcessor(params, fileUtil)
    fileUtil.addExpectedFileWritten(processor.predicateRankingLfFile, expectedSciencePredicateLfOutput)
    fileUtil.onlyAllowExpectedFiles()
    val wordCounts = processor.getWordCountsFromTrainingFile()
    val trainingData = processor.readTrainingData(wordCounts)
    processor.outputPredicateRankingLogicalForms(trainingData)
    fileUtil.expectFilesWritten()
  }

  "outputQueryRankingLogicalForms" should "give the correct output on clueweb input" in {
    val extraParams: JValue = ("training data file" -> clueWebDataFilename)
    val params = baseParams merge extraParams
    val fileUtil = getFileUtil()
    val processor = new TrainingDataProcessor(params, fileUtil)
    fileUtil.addExpectedFileWritten(processor.queryRankingLfFile, expectedClueWebQueryLfOutput)
    val wordCounts = processor.getWordCountsFromTrainingFile()
    val trainingData = processor.readTrainingData(wordCounts)
    val queryIndex = processor.outputJointEntityFile(trainingData)
    processor.outputQueryRankingLogicalForms(trainingData, queryIndex)
    fileUtil.expectFilesWritten()
  }

  it should "give the correct output on sciece input" in {
    val extraParams: JValue = ("training data file" -> scienceDataFilename)
    val params = baseParams merge extraParams
    val fileUtil = getFileUtil()
    val processor = new TrainingDataProcessor(params, fileUtil)
    fileUtil.addExpectedFileWritten(processor.queryRankingLfFile, expectedScienceQueryLfOutput)
    val wordCounts = processor.getWordCountsFromTrainingFile()
    val trainingData = processor.readTrainingData(wordCounts)
    val queryIndex = processor.outputJointEntityFile(trainingData)
    processor.outputQueryRankingLogicalForms(trainingData, queryIndex)
    fileUtil.expectFilesWritten()
  }
}
