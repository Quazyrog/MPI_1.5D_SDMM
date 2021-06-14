#ifndef ZADANIE__DEBUG_HPP
#define ZADANIE__DEBUG_HPP
#include <sstream>

namespace Debug {

constexpr bool ENABLED = true;

template<class TIter>
std::string VectorToString(TIter begin, TIter end)
{
    if (ENABLED) {
        std::stringstream s;
        s << "[";
        bool comma = false;
        for (auto it = begin; it != end; ++it) {
            if (comma)
                s << ", ";
            s << *it;
            comma = true;
        }
        s << "]";
        return s.str();
    }
    return "[...]";
}

template<class T>
std::string VectorToString(const std::vector<T> &vec)
{
    return VectorToString(vec.begin(), vec.end());
}

}

#endif //ZADANIE__DEBUG_HPP
