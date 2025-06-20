#pragma once
#include "alpha_tree/queues/QItem.hpp"
#include "alpha_tree/utility/defines.hpp"

class QuadHeapQueue {
    ImgIdx cursize;
    ImgIdx maxsize;
    QItem *arr;

  public:
    ImgIdx get_cursize() { return cursize; }
    ImgIdx size() { return cursize; }
    ImgIdx sizeMax() { return maxsize; }
    bool empty() { return cursize == 0; }

    QuadHeapQueue(ImgIdx maxsize_in);
    ~QuadHeapQueue();

    void print();

    ImgIdx pop();
    void push(QItem item);
    inline const QItem &top() const { return arr[1]; }
};