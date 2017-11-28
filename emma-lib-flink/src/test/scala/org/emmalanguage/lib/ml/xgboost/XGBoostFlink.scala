/*
 * Copyright © 2017 The Streamline Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.emmalanguage
package lib.ml.xgboost

import api._
import api.flink._
import Meta._

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.SparseVector
import ml.dmlc.xgboost4j.java.{IRabitTracker, RabitTracker => PythonRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}
import org.apache.flink.ml.math.DenseVector

object XGBoostFlink extends FlinkAware {


  def run(input: String, dimension: Int, rounds: Int, params: Map[String, Any]) = {
    withDefaultFlinkEnv(implicit flink => emma.onFlink {

      val trackerConf = TrackerUtils.getDefaultTackerConf
      val tracker = TrackerUtils.startTracker(flink.getParallelism, trackerConf)

      val in = DataBag.readText(input) map {
        line => {
          val splits = line.split(' ')
          val label = splits.head
          val featuresMap = splits.tail
          val features = featuresMap map {
            str =>
              val pair = str.split(':')
              val index = pair(0).toInt - 1
              val value = pair(1).toDouble

              (index, value)
            }
          Some(LabeledVector(label.toDouble, SparseVector.fromCOO(dimension, features)))
        }
      }

      val ds = toDataSet[LabeledVector](in)

      val input: DataBag[((LabeledVector, Long))] =
        fromDataSet(ds.map(new RichMapFunction[LabeledVector, (LabeledVector, Long)]() {
        override def map(value: LabeledVector) = (LabeledPoint(value.label, value.vector.), getRuntimeContext.getIndexOfThisSubtask)
      }).map(new LabelVectorToLabeledPointMapper))

      val model = XGBoost.train(input, tracker, rounds, params)




    })
  }

}

private [xgboost] class LabelVectorToLabeledPointMapper extends RichMapFunction[LabeledVector, LabeledPoint] {
  override def map(x: LabeledVector): LabeledPoint = {
    var index: Array[Int] = Array[Int]()
    var value: Array[Double] = Array[Double]()
    x.vector match {
      case s: SparseVector =>
        index = s.indices
        value = s.data
      case d: DenseVector =>
        val (i, v) = d.toSeq.unzip
        index = i.toArray
        value = v.toArray
    }
    LabeledPoint(x.label.toFloat,
      index, value.seq.map(z => z.toFloat).toArray)
  }
}

/**
  * Rabit tracker configurations.
  *
  * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
  *                                Set timeout length to zero to disable timeout.
  *                                Use a finite, non-zero timeout value to prevent tracker from
  *                                hanging indefinitely (in milliseconds)
  *                                (supported by "scala" implementation only.)
  * @param trackerImpl Choice between "python" or "scala". The former utilizes the Java wrapper of
  *                    the Python Rabit tracker (in dmlc_core), whereas the latter is implemented
  *                    in Scala without Python components, and with full support of timeouts.
  *                    The Scala implementation is currently experimental, use at your own risk.
  */
private [xgboost] case class TrackerConf(workerConnectionTimeout: Long, trackerImpl: String)

private [xgboost] object TrackerUtils {

  /**
    * Get the default TrackerConf object which chooses the python implemented Tracker.
    *
    * @return the TrackerConf with the basic config
    */
  def getDefaultTackerConf: TrackerConf = TrackerConf(0L, "python")

  /**
    * Start the Tracker which is defined in the trackerConf object.
    * It will throw an IllegalArgumentException if can not be started.
    *
    * @param nWorkers is the number of parallelism of the tracked job. The number of trackers
    * @param trackerConf contains the necessary configuration to create the Tracker.
    *                    If it is not presented than the default tracker will be used.
    * @return with the IRabitTracker trait
    */
  def startTracker(nWorkers: Int,
                   trackerConf: TrackerConf = getDefaultTackerConf
                  ): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => new ScalaRabitTracker(nWorkers)
      case "python" => new PythonRabitTracker(nWorkers)
      case _ => new PythonRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

}