package io.ssc.logisticregression.mlclassex2;

import io.ssc.logisticregression.RegularizedLogisticRegression;
import io.ssc.logisticregression.TrainingExample;
import org.junit.Test;

public class Ex2Test {

  @Test
  public void exercise2() {

    TrainingExample[] examples = TrainingData.getMappedTrainingData();

    double alpha = -10;
    double lambda = 0.05;

    RegularizedLogisticRegression classifier = new RegularizedLogisticRegression(examples, alpha, lambda);

    System.out.println("Accuracy before training: " + classifier.accuracy());

    classifier.trainWithBatchGradientDescent(500);

    System.out.println("Accuracy after training: " + classifier.accuracy());
  }

}
