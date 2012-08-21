package io.ssc.logisticregression;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

public class RegularizedLogisticRegression {

  private final TrainingExample[] examples;
  private final double alpha;
  private final double lambda;
  private final int numFeatures;

  private Vector theta;

  public RegularizedLogisticRegression(TrainingExample[] examples, double alpha, double lambda) {
    Preconditions.checkArgument(examples.length > 0);
    this.examples = examples;
    this.alpha = alpha;
    this.lambda = lambda;

    numFeatures = examples[0].x().size();

    theta = zeros(numFeatures);
  }

  public double accuracy() {
    int numCorrect = 0;

    for (TrainingExample example : examples) {
      double prediction = hypothesis(example.x());

      if ((example.y() == 1 && prediction > 0.5) || (example.y() == 0 && prediction <= 0.5)) {
        numCorrect++;
      }
    }

    return (double) numCorrect / examples.length;
  }

  public double hypothesis(Vector x) {
    return 1d / (1d + Math.exp(-1d * theta.dot(x)));
  }

  public void trainWithBatchGradientDescent(int numIterations) {
    for (int n = 0; n < numIterations; n++) {
      batchGradientDescentSinglePass();
    }
  }

  public Vector theta() {
    return theta;
  }

  private void batchGradientDescentSinglePass() {

    double m = examples.length;
    double alphaDivM = alpha / m;


    Vector gradients = zeros(28);

    for (TrainingExample example : examples) {
      Vector partialGradient = partialGradient(theta, example.x(), example.y(), m, lambda);

      partialGradient.assign(Functions.MULT, alphaDivM);

      gradients.assign(partialGradient, Functions.MINUS);
    }

    theta = theta.minus(gradients);
  }

  private Vector zeros(int cardinality) {
    return new DenseVector(cardinality);
  }

  private Vector partialGradient(Vector theta, Vector x, double y, double m, double lambda) {

    Vector regularization = theta.times(lambda / m);
    regularization.set(0, 0);

    double hypothesisMinusLabel = hypothesis(x) - y;

    Vector partialGradient = x.times(hypothesisMinusLabel).minus(regularization);

    return partialGradient;
  }
}
