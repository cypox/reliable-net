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

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  caffe::Blob<float> input_data(1, 1, 103, 1);

  std::vector<float> input_vec(103, 0.0);
  input_data.set_cpu_data(&input_vec[0]);

  input_layer->Reshape(1, 1, 103, 1);
  net_->Reshape();

  net_->Forward();

  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  std::vector<float> output(begin, end);

  for ( int i = 0 ; i < output.size() ; ++ i )
  {
    std::cout << output[i] << std::endl;
  }

  return 0;
}
