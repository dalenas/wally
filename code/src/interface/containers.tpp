using namespace Wally;

template<typename T>
Vector<T>::Vector(std::size_t s)
    : size_(s) { data_.reserve(size_); }

template<typename T>
Vector<T>::Vector(std::size_t s, T init)
    : size_(s) { data_.assign(size_, init); }

template<typename T>
Vector<T>::Vector(const std::vector<std::vector<T>>& M) 
    : size_(M.size()) {
    for(std::size_t i = 0; i < size_; ++i)
        data_[i] = M[i][0];
}

template<typename T>
Vector<T>::Vector(const Vector<T>& v) 
    : size_(v.size()) {
    data_.reserve(v.size());
    for(std::size_t i = 0; i < size_; ++i)
        data_[i] = v(i);
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& v) {
    this->size_ = v.size();

    data_.reserve(v.size());
    for(std::size_t i = 0; i < v.size(); ++i)
        data_[i] = v(i);

    return *this;
}

template<typename T>
T& Vector<T>::operator()(std::size_t i) {
    return data_[i];
}

template<typename T>
const T& Vector<T>::operator()(std::size_t i) const {
    return data_[i];
}

template<typename T>
std::size_t Vector<T>::size() const {
    return size_;
}

template<typename T>
T* Vector<T>::data() {
    return data_.data();
}

template<typename T>
const T* Vector<T>::data() const {
    return data_.data();
}

template<typename U>
Matrix<U>::Matrix(std::size_t r, std::size_t c)
    : rows_(r), cols_(c), axis_(Major::row) { data_.reserve(rows_ * cols_); }

template<typename U>
Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a)
    : rows_(r), cols_(c), axis_(a) { data_.reserve(rows_ * cols_); }

template<typename U>
Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a, U init)
    : rows_(r), cols_(c), axis_(a) { data_.assign(rows_ * cols_, init); }

template<typename U>
Matrix<U>::Matrix(const std::vector<std::vector<U>>& M) 
    : rows_(M[0].size()), cols_(M.size()) {
    axis_ = rows_ >= 8 ? Major::row : Major::col;               // temporary while I figure out the best way to decide
    
    if(axis_ == Major::row) {
        for(std::size_t i = 0; i < rows_; ++i) {
            for(std::size_t j = 0; j < cols_; ++j)
                data_[j + i*cols_] = M[i][j];
        }
    } else {
        for(std::size_t j = 0; j < cols_, ++j) {
            for(std::size_t i = 0; i < rows_; ++i)
                data_[i + j*rows_] = M[i][j];
        }
    }
}

template<typename U>
Matrix<U>::Matrix(const Matrix<U>& X)
    : rows_(X.rows()), cols_(X.cols()), axis_(X.axis()) {
    data_.reserve(X.size());
    const U* const x_data = X.data();
    for(std::size_t i = 0; i < X.size(); ++i)
        data_[i] = x_data[i];
}

template<typename U>
Matrix<U>& Matrix<U>::operator=(const Matrix<U>& X) {
    rows_ = X.rows();
    cols_ = X.cols();
    axis_ = X.axis();

    data_.reserve(X.size());
    const U* const x_data = X.data();
    for(std::size_t i = 0; i < X.size(); ++i)
        data_[i] = x_data[i];

    return *this;
}

template<typename U>
U& Matrix<U>::operator()(std::size_t row, std::size_t col) {
    if(axis_ == Major::col)
        return data_[row + col*rows_];
    return data_[col + row*cols_];
}

template<typename U>
const U& Matrix<U>::operator()(std::size_t row, std::size_t col) const {
    if(axis_ == Major::col)
        return data_[row + col*rows_];
    return data_[col + row*cols_];
}

template<typename U>
std::size_t Matrix<U>::rows() const {
    return rows_;
}

template<typename U>
std::size_t Matrix<U>::cols() const {
    return cols_;
}

template<typename U>
std::size_t Matrix<U>::size() const {
    return rows_ * cols_;
}

template<typename U>
Major Matrix<U>::axis() const {
    return axis_;
}


template<typename U>
U* Matrix<U>::data() {
    return data_.data();
}

template<typename U>
const U* Matrix<U>::data() const {
    return data_.data();
}