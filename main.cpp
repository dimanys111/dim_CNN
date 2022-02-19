#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>  /* The numerical optimization library that mlpack uses */ 

using namespace mlpack;
using namespace mlpack::ann;

// Namespace for the armadillo library(linear algebra library).
using namespace arma;
using namespace std;

// Namespace for ensmallen.
using namespace ens;

arma::Row<size_t> getLabels(arma::mat predOut)
{
  arma::Row<size_t> predLabels(predOut.n_cols);
  for(arma::uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max() + 1;
  }
  return predLabels;
}

constexpr double RATIO = 0.1; // ratio to divide the data in train and val set.
constexpr int MAX_ITERATIONS = 0; // set to zero to allow infinite iterations.
constexpr double STEP_SIZE = 1.2e-3;// step size for Adam optimizer.
constexpr int BATCH_SIZE = 50;
constexpr size_t EPOCH = 2;

int main()
{
  mat tempDataset;
  data::Load("train.csv", tempDataset, true);

  mat tempTest;
  data::Load("test.csv", tempTest, true);

  mat dataset = tempDataset.submat(0, 1, tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  mat test = tempTest.submat(0, 1, tempTest.n_rows - 1, tempTest.n_cols - 1);

  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
  const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);
  const mat testX = test.submat(1, 0, test.n_rows - 1, test.n_cols - 1);
  const mat trainY = train.row(0) + 1;
  const mat validY = valid.row(0) + 1;
  const mat testY = test.row(0) + 1;

  FFN<NegativeLogLikelihood<>, RandomInitialization> model;

  model.Add<Convolution<>>(1,  // Number of input activation maps.
                          6,  // Number of output activation maps.
                          5,  // Filter width.
                          5,  // Filter height.
                          1,  // Stride along width.
                          1,  // Stride along height.
                          0,  // Padding width.
                          0,  // Padding height.
                          28, // Input width.
                          28  // Input height.
  );

  model.Add<ReLULayer<>>();

  model.Add<MaxPooling<>>(2, // Width of field.
                          2, // Height of field.
                          2, // Stride along width.
                          2, // Stride along height.
                          true);

  model.Add<Convolution<>>(6, // Number of input activation maps.
                          16, // Number of output activation maps.
                          5, // Filter width.
                          5, // Filter height.
                          1, // Stride along width.
                          1, // Stride along height.
                          0, // Padding width.
                          0, // Padding height.
                          12, // Input width.
                          12  // Input height.
  );

  model.Add<ReLULayer<>>();
                          
  model.Add<MaxPooling<>>(2, 2, 2, 2, true);
                          
  model.Add<Linear<>>(16 * 4 * 4, 10);
                          
  model.Add<LogSoftMax<>>();  

  ens::Adam optimizer(
    STEP_SIZE,  // Step size of the optimizer.
    BATCH_SIZE, // Batch size. Number of data points that are used in each iteration.
    0.9,        // Exponential decay rate for the first moment estimates.
    0.999, // Exponential decay rate for the weighted infinity norm estimates.
    1e-8,  // Value used to initialise the mean squared gradient parameter.
    MAX_ITERATIONS, // Max number of iterations.
    1e-8, // Tolerance.
    true);

  model.Train(trainX,
            trainY,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss(EPOCH),
            ens::EarlyStopAtMinLoss(
                [&](const arma::mat& /* param */)
                {
                  double validationLoss = model.Evaluate(validX, validY);
                  std::cout << "Validation loss: " << validationLoss
                      << "." << std::endl;
                  return validationLoss;
                }));

  mat predOut;
  model.Predict(trainX, predOut);
  arma::Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy = arma::accu(predLabels == trainY) / ( double )trainY.n_elem * 100;
  model.Predict(validX, predOut);
  predLabels = getLabels(predOut);
  double validAccuracy = arma::accu(predLabels == validY) / ( double )validY.n_elem * 100;
  std::cout << "Accuracy: train = " << trainAccuracy << "%,"<< "\t valid = " << validAccuracy << "%" << std::endl;

  mat testPredOut;
  model.Predict(testX,testPredOut);
  arma::Row<size_t> testPred = getLabels(testPredOut);
  double testAccuracy = arma::accu(testPredOut == testY) /( double )trainY.n_elem * 100;
  std::cout<<"Test Accuracy = "<< testAccuracy;
  
}