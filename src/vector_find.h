/**
 * \file
 * \brief Helper functions to find elements within a vector
 */
#ifndef VECTOR_FIND_H
#define VECTOR_FIND_H

#include <vector>
#include <stdexcept>
#include <algorithm>

/**
 * \brief Finds the maximum element of a (templated) list and returns its content
 * \tparam T the data type (e.g. float)
 * \param list the vector of values of type T
 * \return the maximum value, with type T
 * \throw runtime error if list is empty
 */
template<typename T>
T findMax(const std::vector<T> &list)
{
    if(list.size()==0)
    {
        throw std::runtime_error("Empty list");
    }
    else
    {
        return *std::max_element(list.begin(),list.end());
    }
}

/**
 * \brief Finds the minimum element of a (templated) list and returns its content
 * \tparam T the data type (e.g. float)
 * \param list the vector of values of type T
 * \return the minimum value, with type T
 * \throw runtime error if list is empty
 */
template<typename T>
T findMin(const std::vector<T> &list)
{
    if(list.size()==0)
    {
        throw std::runtime_error("Empty list");
    }
    else
    {
        return *std::min_element(list.begin(),list.end());
    }
}

/**
 * \brief Finds the index of the first element in an ordered (templated) list larger than an input value
 * \note If value is smaller than the smallest element, then 0 is returned
 * \tparam T the data type (e.g. float)
 * \param orderedList the vector of ordered values of type T
 * \param value the threshold value, of type T
 * \return the index (position in the array) of the first element above the threshold
 * \todo use std::set instead
 */
template<typename T>
int findFirstLargerOrdered(const std::vector<T> &orderedList, T value)
{
    int upper = int(orderedList.size()-1);
    int lower = 0;

    if (orderedList.back() <= value) {
        return upper;
    }
    else if (orderedList.front() > value) {
        return 0;
    }
    else {
        while (upper - lower > 1) {
            int pivot = (upper + lower) / 2;
            // Checking and replacing lower before upper ensures finds last index if several values are the same
            if (orderedList[pivot] <= value) {lower = pivot;}
            else {upper = pivot;}
        }
        return lower+1;
    }
}

/**
 * \brief Finds the index of the last element in an ordered (templated) list smaller or equal than an input value
 * \note If value is smaller than the smallest element, then -1 is returned
 * \tparam T the data type (e.g. float)
 * \param orderedList the vector of ordered values of type T
 * \param value the threshold value, of type T
 * \return the index (position in the array) of the last element below or equal to the threshold
 * \todo use std::set instead
 */
template<typename T>
int findLastSmallerOrEqOrdered(const std::vector<T> &orderedList, T value)
{
    int upper = int(orderedList.size()-1);
    int lower = 0;

    if (orderedList.back() <= value) {
        return upper;
    }
    else if (orderedList.front() > value) {
        return -1;
    }
    else {
        while (upper - lower > 1) {
            int pivot = (upper + lower) / 2;
            // Checking and replacing lower before upper ensures finds last index if several values are the same
            if (orderedList[pivot] <= value) {lower = pivot;}
            else {upper = pivot;}
        }
        return lower;
    }
}

/**
 * \brief Finds the decimal index of a value of templated type TIn by interpolating between the nearest higher and lower values
 * \note TOut should not be of integer type
 * (in this case runtime errors are expected if TIn is of floating point type,
 * otherwise the same result as from findLastSmallerOrEqOrdered would be expected)
 * \tparam TIn the input data type (e.g. int)
 * \tparam TOut the output data type (e.g. float)
 * \note If value is smaller/larger than than the smallest/largest element in orderedList, the first/last index is returned
 * \param orderedList the vector of ordered values of type Tin
 * \param value the threshold value, of type TOut
 * \return the decimal index as type TOut (interpolated position in the array) of the last element below or equal to the threshold
 * \todo use std::set instead
 */
template <typename TIn, typename TOut>
TOut findDecimalOrdered(const std::vector<TIn> &orderedList, TOut value)
{
    // Ensure last idx returned if all items in list the same and equal to value by checking last element before first
    if (value >= TOut(orderedList.back())) {
        return TOut(orderedList.size()-1);
    }
    else if (value < TOut(orderedList.front())) {
        return TOut(0.0);
    }
    else {
        unsigned int floorIdx = findLastSmallerOrEqOrdered<TIn> (orderedList, TIn(value));
        TOut corr = (value - TOut(orderedList[floorIdx])) / TOut(orderedList[floorIdx+1] - orderedList[floorIdx]);
        return TOut(floorIdx) + corr;
    }
}

#endif // VECTOR_FIND_H
