/**
 * \file
 * \brief Helper functions for vector interpolation
 */
#ifndef VECTOR_INTERPOLATE_H
#define VECTOR_INTERPOLATE_H

#include <vector>
#include <cmath>

/**
 * \brief Linearly interpolates into list at decimal index idx
 * \note If idx is smaller/larger than the first(0)/last index, the first/last value is returned
 * \param list the array of values, of type TIn
 * \param idx the decimal index, of type TOut
 * \return the interpolated value, of type TOut
 */
template <typename TIn, typename TOut>
TOut vectorInterpolate(const std::vector<TIn> &list, TOut idx)
{
    if (idx <= TOut(0.0)) {return TOut(list.front());}
    else if (idx >= TOut(list.size()-1)) {return TOut(list.back());}
    else {
        TOut intPart;
        TOut decimals = std::modf(idx, &intPart);
        unsigned int floorIdx = static_cast<unsigned int>(intPart);
        TOut corr = TOut(list[floorIdx + 1] - list[floorIdx]) * decimals;
        return list[floorIdx] + corr;
    }
}

#endif // VECTOR_INTERPOLATE_H
