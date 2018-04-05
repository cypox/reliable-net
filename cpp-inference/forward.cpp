#include <caffe/caffe.hpp>
#include <string>
#include <vector>


int main(int argc, char** argv) {

  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel" << std::endl;
    return -1;
  }

  ::google::InitGoogleLogging(argv[0]);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);

  std::string model_file   = argv[1];
  std::string trained_file = argv[2];

  std::shared_ptr<caffe::Net<float> > net_;


  /* Load the network. */
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  boost::shared_ptr<caffe::Blob<float> > input_layer = net_->blob_by_name("input");
  //caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  float* input_layer_data = input_layer->mutable_cpu_data();
  for ( int i = 0 ; i < 100 ; ++ i )
  {
    input_layer_data[i] = 1.0;
  }

  input_layer->Reshape(1, 1, 100, 1);
  net_->Reshape();

  net_->Forward();

  /* Copy the output layer to a std::vector */
  boost::shared_ptr<caffe::Blob<float> > input_blop = net_->blob_by_name("input");
  const float* begin1 = input_blop->cpu_data();
  const float* end1 = begin1 + 100;
  std::vector<float> input(begin1, end1);

  /* Copy the output layer to a std::vector */
  boost::shared_ptr<caffe::Blob<float> > output_layer_by_name = net_->blob_by_name("output");
  const float* begin = output_layer_by_name->cpu_data();
  const float* end = begin + 100;
  std::vector<float> result(begin, end);

  for ( int i = 0 ; i < result.size() ; ++ i )
  {
    std::cout << input[i] << " " << result[i] << std::endl;
  }

  /* Copy the output layer to a std::vector */
  /* OLD VERSION (WITHOUT USING LAYER NAMES)
  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  //the end pointer should not use the number of channels since this can be modified prior to the output layer via a reshape layer
  //const float* end = begin + output_layer->channels();
  const float* end = begin + 100;
  std::vector<float> output(begin, end);

  std::cout << "output size : " << output.size() << std::endl;

  for ( int i = 0 ; i < output.size() ; ++ i )
  {
    std::cout << output[i] << std::endl;
  }
  //*/

  return 0;
}
