#include "linear_regression.h"

class PolynomialRegression : public LinearRegression {
private:
    int degree;

    int num_features(const int&);
    int choose(const int&, const int&);
    std::vector<std::vector<int>> pascal_matrix(const int&, const int&);
public:
    PolynomialRegression(const int&);
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>&);
    ~PolynomialRegression() = default;
};