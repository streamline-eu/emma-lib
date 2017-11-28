/*
 * Copyright © 2017 TU Berlin (emma@dima.tu-berlin.de)
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

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.java.IRabitTracker
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.EvalTrait
import ml.dmlc.xgboost4j.scala.ObjectiveTrait
import ml.dmlc.xgboost4j.scala.{XGBoost => XGBoostScala}

@emma.lib
object XGBoost {

  def train(
    instances: DataBag[(LabeledPoint, Long)],
    tracker: IRabitTracker,
    round: Int,
    params: Map[String, Any],
    obj: ObjectiveTrait = null,
    eval: EvalTrait = null,
  ): Booster = {

    (
      for (Group(pid, partition: DataBag[(LabeledPoint, Long)]) <- instances groupBy { _._2 } ) yield {

        tracker.getWorkerEnvs.put("DMLC_TASK_ID", pid.toString)
        val trainMat: DMatrix = new DMatrix(partition.map(_._1).collect().toIterator)
        Rabit.init(tracker.getWorkerEnvs)

        val booster = XGBoostScala.train(
          trainMat, params, round, watches = List("train" -> trainMat).toMap, obj, eval)

        Rabit.shutdown()
        trainMat.delete()

        booster
      }
    ).collect().head


  }

}

