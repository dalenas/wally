namespace Wally {
    template<typename T>
    Vector<T>::Vector(std::size_t s)
        : size_(s) { data_.resize(size_); }

    template<typename T>
    Vector<T>::Vector(std::size_t s, T init)
        : size_(s) {
        data_.reserve(size_);
        for(std::size_t i = 0; i < size_; ++i)
            data_.push_back(init);
    }

    template<typename T>
    Vector<T>::Vector(const std::vector<std::vector<T>>& M) 
        : size_(M.size()) {
        data_.reserve(M.size());
        for(std::size_t i = 0; i < size_; ++i)
            data_.push_back(M[i][0]);
    }

    template<typename T>
    Vector<T>::Vector(const Vector<T>& v) 
        : size_(v.size()) {
        data_.reserve(v.size());
        for(std::size_t i = 0; i < size_; ++i)
            data_.push_back(v(i));
    }

    template<typename T>
    Vector<T>& Vector<T>::operator=(const Vector<T>& v) {
        this->size_ = v.size();

        data_.reserve(v.size());
        for(std::size_t i = 0; i < v.size(); ++i)
            data_.push_back(v(i));

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
        : rows_(r), cols_(c) {
        axis_ = cols_ >= 8 ? Major::col : Major::row;
        data_.reserve(rows_ * cols_);
    }

    template<typename U>
    Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a)
        : rows_(r), cols_(c), axis_(a) { data_.reserve(rows_*cols_); }

    template<typename U>
    Matrix<U>::Matrix(std::size_t r, std::size_t c, Major a, U init)
        : rows_(r), cols_(c), axis_(a) {
        const std::size_t size_ = rows_*cols_;
        data_.resize(size_);
        for(std::size_t i = 0; i < size_; ++i)
            data_[i] = init;
    }

    template<typename U>
    Matrix<U>::Matrix(const std::vector<std::vector<U>>& M) 
        : rows_(M.size()), cols_(M[0].size()) {
        axis_ = cols_ >= 8 ? Major::col : Major::row;               // temporary while I figure out the best way to decide
        
        data_.resize(rows_*cols_);
        if(axis_ == Major::row) {
            for(std::size_t i = 0; i < rows_; ++i) {
                std::size_t row_num = i*cols_;
                for(std::size_t j = 0; j < cols_; ++j)
                    data_[j + row_num] = M[i][j];
            }
        } else {
            for(std::size_t j = 0; j < cols_; ++j) {
                std::size_t col_num = j*rows_;
                for(std::size_t i = 0; i < rows_; ++i)
                    data_[i + col_num] = M[i][j];
            }
        }
    }

    template<typename U>
    Matrix<U>::Matrix(const Matrix<U>& X)
        : rows_(X.rows()), cols_(X.cols()), axis_(X.axis()) {
        data_.resize(X.size());
        const U* const x_data = X.data();
        for(std::size_t i = 0; i < X.size(); ++i)
            data_[i] = x_data[i];
    }

    template<typename U>
    Matrix<U>& Matrix<U>::operator=(const Matrix<U>& X) {
        rows_ = X.rows();
        cols_ = X.cols();
        axis_ = X.axis();

        data_.resize(X.size());
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
}