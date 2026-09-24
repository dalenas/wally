#include "preprocessing/dataset.h"

namespace Wally {
    Matrix<std::string> read_csv(const std::string& filename) {
        std::ifstream file(filename);
        if(!file)
            std::cerr << "Could not open file." << std::endl;

        std::vector<std::vector<std::string>> raw_data;
        std::string line;
        std::getline(file, line);
        while(std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream stream(line);
            std::string point;

            while(std::getline(stream, point, ','))
                row.push_back(point);
            raw_data.push_back(row);
        }

        Matrix<std::string> data(raw_data);

        file.close();
        return data;
    }

    Matrix<float> cvt_float(const Matrix<std::string>& data, const std::size_t col) {
        const std::size_t N = data.rows();

        Matrix<float> cvt_data(N, 1);
        for(std::size_t i = 0; i < N; ++i)
            cvt_data(i, 0) = std::stof(data(i, col));

        return cvt_data;
    }

    Matrix<float> cvt_float(const Matrix<std::string>& data, const std::vector<std::size_t>& cols) {
        const std::size_t N = data.rows();
        const std::size_t M = cols.size();

        Matrix<float> cvt_data(N, M);
        for(std::size_t j = 0; j < M; ++j) {
            std::size_t col = cols[j];
            for(std::size_t i = 0; i < N; ++i)
                cvt_data(i, j) = std::stof(data(i, col));
        }
        
        return cvt_data;
    }

    Matrix<float> get_features(const Matrix<float>& data, const std::vector<std::size_t>& cols) {
        const std::size_t N = data.rows();
        const std::size_t M = cols.size();

        Matrix<float> X(N, M);
        for(std::size_t j = 0; j < M; ++j) {
            std::size_t col = cols[j];
            for(std::size_t i = 0; i < N; ++i)
                X(i, j) = data(i, col);
        }

        return X;
    }

    Vector<float> get_target(const Matrix<float>& data, const std::size_t col) {
        const std::size_t N = data.rows();

        Vector<float> y(N);
        for(std::size_t i = 0; i < N; ++i)
            y(i) = data(i, col);

        return y;
    }
};