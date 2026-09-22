#ifndef DATASET_H
#define DATASET_H

#include <iostream>

#include "abstract/traits.h"

namespace Wally {
    template<typename Container>
    Container read_csv(const std::string& filename) {
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

        if(raw_data[0].size() == 1)
            Vector<std::string> data(raw_data);
        else
            Matrix<std::string> data(raw_data);

        file.close();
        return data;
    };
}

#endif