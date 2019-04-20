#include <iostream>
#include <armadillo>
#include <gamelogic.hpp>
#include <NeuralNetwork.hpp>
#include <chrono> 
#include <fstream>

int main(int argc, const char **argv) {
    // arma::arma_rng::set_seed_random();
    Params params;
    params.in_h = 2;
    params.in_w = 2;
    params.hlc  = 2;
    params.npl  = 4;

    NeuralNetwork pAn(params);

    arma::Mat<float> save1 = arma::randu<arma::Mat<float>>(3, 3);
    arma::Mat<float> save2 = arma::randu<arma::Mat<float>>(3, 3);

    std::ofstream out_file;
    out_file.open("test.bin", std::ios::out | std::ios::binary);

    save1.save(out_file);
    save2.save(out_file);

    out_file.close();

    arma::Mat<float> load1;
    arma::Mat<float> load2;

    std::ifstream in_file;
    in_file.open("test.bin", std::ios::in | std::ios::binary);

    load1.load(in_file);
    load2.load(in_file);

    std::cout << "Save1:\n" << save1 << std::endl;
    std::cout << "Save2:\n" << save2 << std::endl;
    std::cout << "load1:\n" << load1 << std::endl;
    std::cout << "load2:\n" << load2 << std::endl;

    return 0;
}