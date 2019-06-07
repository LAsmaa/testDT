import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import java.io.FileWriter
import breeze.linalg.max
import scala.collection.mutable.ArrayBuffer




object test {

  def main(args: Array[String]){

    /** ----------------------------------------------
      *|            CHANGE THIS PATH!!!!!             |
      * ----------------------------------------------
      */

    val path = "Relativ_path/nursery.txt"


    val conf = new SparkConf().setAppName("RddToDataframe").setMaster("local[*]")
    val sc = new SparkContext(conf)

    var text = sc.textFile(path)
    var ligne = text.map(line => line.split(","))

    var impurity = Entropy.calculate(ligne, Array())

    var learner = new Node(ligne, 0,
      "Root", 10, 0.3, Array())

    learner.run()


  }

}


object Entropy{


  def calculate(rawRdd: RDD[Array[String]], toIgnoreAttributes: Array[Int]): Int= {


    //Todo: Compute a fct that calculates the best split more efficiently, without having an array of all the entropies
    var start: Int = 1
    var stop: Int = rawRdd.first().length
    var labelIndex: Int = 0
    start = 0
    stop -=1
    labelIndex = stop

    var bestImpurity = new Array[String](2)
    val labelRdd = rawRdd.map(line => line(labelIndex)).distinct
    val labelSize = labelRdd.count().toInt
    val labelList = rawRdd.map(line => line(labelIndex)).distinct.take(labelSize)
    var impurities = new Array[Double](stop)

    for (i <- start until stop){
      if(!toIgnoreAttributes.contains(i)){
        val helper01 = rawRdd.map(line => (line(i), line (labelIndex)))
        val helper02 = helper01.reduceByKey((a,b) => a+","+b)
        val helper03 = helper02.map(line => (line._1, line._2.split(",")))
        val attributeSize = rawRdd.map(line => line(i)).distinct.count.toInt
        var oneValueStats = new Array[Int](attributeSize*labelSize)

        var cpt = 0
        for (j <- 0 until attributeSize){

          val helper04 = helper03.zipWithIndex.filter(_._2==j).map(_._1).first._2
          val helper05 = helper04.groupBy(identity).mapValues(_.size)

          var comptage = 0
          for(k <- labelList){

            if (helper05.contains(k))
              comptage =  helper05(k)
            else
              comptage = 0

            oneValueStats(cpt) = comptage

            //System.out.println("here it is " + comptage + " For label " + k + " And attribute " + i + " Qui est " + oneValueStats(cpt) + " Et cpt rah  " + cpt)
            cpt += 1
          }
        }
        var result = calculateOneAttribute(oneValueStats, labelList.length)
        impurities(i) = result
      }else
        impurities(i) = 2 //Impurity f l'entropy toujours entre 0 et 1
    }
    return bestSplit(impurities)
  }

  def log2(x: Double) = scala.math.log(x) / scala.math.log(2)

  /**
    * :: DeveloperApi ::
    * information calculation for multiclass classification
    * @param counts Array[Double] with counts for each label
    * @param totalCount sum of counts for all labels
    * @return information value, or 0 if totalCount = 0
    */

  def calculateOneValue(counts: Array[Int], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val numClasses = counts.length
    var impurity = 0.0
    var classIndex = 0
    while (classIndex < numClasses) {
      val classCount = counts(classIndex)
      if (classCount != 0) {
        val freq = classCount / totalCount
        impurity -= freq * log2(freq)
      }
      classIndex += 1
    }
    impurity
  }



  /**
    * :: AsmaaApi ::
    *
    * @param calculatedList Array[Double] of calculatedImpurityOfValue
    * @param label Number of classes
    * @return the Total entropy of the
    */

  def calculateOneAttribute(calculatedList: Array[Int], label: Int): Double ={

    var totalSum = calculatedList.sum
    var cpt:Int = 0
    var oneValueStats = new Array[Int](label)
    var impurity: Double = 0

    while (cpt < calculatedList.length){
      for(j <- 0 until label){
        oneValueStats(j) = calculatedList(cpt)
        cpt += 1
      }

      var somme:Double = oneValueStats.sum

      var freq:Double = somme / totalSum
      var calculated = calculateOneValue(oneValueStats, somme)
      //System.out.println(f"The frequence if $freq.5f and the result is $calculated%.5f")
      impurity += (freq*calculated)

    }

    //System.out.println(f"F tali voilà => $impurity%.5f")
    impurity
  }

  /**
    * :: AsmaaApi ::
    * DO NOT FORGET TO CHANGE THIS ACCORDING TO YOUR IMPURITY
    *         MIN OR MAX YOU CHOOSE
    * @param calculatedList Array[Double] with entropies results
    * @return The index of the Max/Min (depends on the Impurity choosen
    */

  def bestSplit(calculatedList: Array[Double]): Int = {
    var indexOfTheBest = minIndex(calculatedList)
    return indexOfTheBest
  }

  def minIndex (a: Array[Double]) : Int= {
    var minimum:Double = a.min
    var minIndex = a.indexOf(minimum)
    minIndex
  }


}

class EntropyCalculator( val raw: RDD[Array[String]], val toIgnoreAttributes: Array[Int]) {


  /**
    * Calculate the impurity from the stored sufficient statistics.
    */
  def calculate(): Int = Entropy.calculate(raw, toIgnoreAttributes)

  def isOnAttribute(): Boolean = true

}

class Node(rawData : RDD[Array[String]], xlevel: Int, xnodeID: String, xMaxLevel: Int, minImpurity: Double, toIgnoreAttributes: Array[Int]) {

  var maxLevel: Int = xMaxLevel
  var level: Int = xlevel +1
  var nodeID: String = xnodeID
  var predict: String = "#None#"
  var impurity: Double = 1
  var isLeaf: Boolean = false
  var children: Array[Node] = null


  /**Function definition**/

  //To run the learning process
  def run(): Unit ={
    var rules: String = nodeID+","

    if(shouldSplit()){
      var bestSplit = chooseSplit(rawData, toIgnoreAttributes)
      rules += splitting(bestSplit)

    }else{
      rules += "#None#,#None#,#None#," + getLabel(rawData)
    }
  }

  def chooseSplit(raw: RDD[Array[String]], toIgnoreAttributes: Array[Int]): Array[String] = {

    val impuritycalculator = new EntropyCalculator(raw, toIgnoreAttributes)
    val onAttribute = impuritycalculator.isOnAttribute()
    var value: String = "#None#"
    val bestSplit = impuritycalculator.calculate()

    value = impuritycalculator.calculate().toString

    Array(bestSplit.toString, value)
  }

  def shouldSplit(): Boolean ={
    if((level < maxLevel || maxLevel == -1) && impurity > minImpurity){
      if(checkData(rawData))
        return true
      else {
        isLeaf = true
        return false
      }
    }
    else {
      isLeaf = true
      return false
    }
  }

  def checkData(givenRDD: RDD[Array[String]]): Boolean={
    if (givenRDD.isEmpty())
      return false
    val labelIndex = givenRDD.first().length -1
    val labels = givenRDD.map(line => line(labelIndex)).distinct.count
    if (labels > 1 && givenRDD.count() > 2)
      return true
    else
      return false
  }

  def onAttributeSplit(attributeIndex: Int): String={
    var result = ""
    val values: Array[String] = RddProcesser.findValues(rawData, attributeIndex)
    var cpt = 0
    var childrenList = ArrayBuffer[Node]()
    val newToIgnore = toIgnoreAttributes ++ Array(attributeIndex)

    for(value <- values){
      val splittedRdd = RddProcesser.horizontaSplittingWithFilter(rawData, value, attributeIndex)
      var child = new Node(splittedRdd, level, nodeID+">"+cpt, xMaxLevel, minImpurity, newToIgnore)
      childrenList += child
      cpt += 1
      result += value + "##-##" + child.nodeID
      if (values.indexOf(value) != values.length-1)
        result += "##/##"
    }
    if(childrenList.nonEmpty){
      for(child <- childrenList)
        child.run()

      children = childrenList.toArray
      result
    } else "#None#,#None#"
  }

  def splitting(bestSplit: Array[String]):String = {

    val separator: String = ","
    val attributeIndex = bestSplit(0).toInt
    var rules: String = attributeIndex + separator
    if (attributeIndex >= 0){
      rules += bestSplit(1) + separator
      rules += onAttributeSplit(attributeIndex)
    } else
      rules += "#None#,#None#"
    rules += separator +  getLabel(rawData)
    rules
  }

  def getLabel(splittedRdd: RDD[Array[String]]): String = {
    var labelIndex = splittedRdd.first().size -1

    var labels = splittedRdd.map(line => line(labelIndex))

    return labels.max()

  }

}

object RddProcesser {

  def horizontaSplittingWithFilter(raw: RDD[Array[String]], filter: String, attributeIndex: Int): RDD[Array[String]]={
    val splittedRdd = raw.filter(line => line(attributeIndex).contains(filter))

    splittedRdd
  }

  def findValues(raw: RDD[Array[String]], attributeIndex: Int): Array[String] = {
    val values = raw.map(line => line(attributeIndex)).distinct
    values.collect()
  }
}


