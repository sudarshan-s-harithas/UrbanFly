#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <vector>
#include <array>

std::vector<double> read_vector(std::string file_path)
{
    std::ifstream file(file_path);
    std::string line;
    std::vector<double> output;

    std::getline(file, line);
    std::stringstream line_stream(line);

    double value;
    while(line_stream >> value)
        output.push_back(value);

    file.close();
    return output;
}

void print_vector(std::string title, std::vector<double> input)
{
    std::cout << title << ": ";
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

void sample_n_values(int n, std::vector<double> means, std::vector<double> covariances, std::vector<double> weights)
{
    std::ofstream samples_file("samples.txt");
    std::random_device rd{};
    std::mt19937 gen{rd()};

    using normal_dist   = std::normal_distribution<>;
    using discrete_dist = std::discrete_distribution<std::size_t>;

    auto G = std::array<normal_dist, 4>{
        normal_dist{means[0], covariances[0]}, // mean, stddev of G[0]
        normal_dist{means[1], covariances[1]}, // mean, stddev of G[1]
        normal_dist{means[2], covariances[2]} , // mean, stddev of G[2]
        normal_dist{means[3], covariances[3]}  // mean, stddev of G[3]
    };

    auto w = discrete_dist{
        weights[0], // weight of G[0]
        weights[1], // weight of G[1]
        weights[2],  // weight of G[2]
        weights[3]  // weight of G[2]
    };

    for (int i = 0 ; i < n; i++){
        auto index = w(gen);
        auto temp_noise_val = G[index](gen);

        samples_file << std::to_string(temp_noise_val) << std::endl;
    }
    samples_file.close();
    std::cout << "noise values are written to file" << std::endl;
}

int main(int argc, char** argv)
{
    std::string gmm_means_file("./minihattan/GMM_4_means.txt");
    std::string gmm_covariances_file("./minihattan/GMM_4_covariances.txt");
    std::string gmm_weights_file("./minihattan/GMM_4_weights.txt");

    std::vector<double> means = read_vector(gmm_means_file);
    std::vector<double> covariances = read_vector(gmm_covariances_file);
    std::vector<double> weights = read_vector(gmm_weights_file);
    
    std::cout << "Sampling N values from the GMM with " << std::endl;
    print_vector("Means", means);
    print_vector("Covariances", covariances);
    print_vector("Weights", weights);

    sample_n_values(std::atoi(argv[1]), means, covariances, weights);

    return 0;
}
