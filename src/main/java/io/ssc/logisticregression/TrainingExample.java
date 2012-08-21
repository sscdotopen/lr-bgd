package io.ssc.logisticregression;

import org.apache.mahout.math.Vector;

public class TrainingExample {

  private final Vector x;
  private final double y;

  public TrainingExample(Vector x, double y) {
    this.x = x;
    this.y = y;
  }

  public Vector x() {
    return x;
  }

  public double y() {
    return y;
  }
}
