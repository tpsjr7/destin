#ifndef DESTINTREEITERATORCALLBACK_H
#define DESTINTREEITERATORCALLBACK_H

class Node;

class DestinTreeIteratorCallback
{
public:
    virtual ~DestinTreeIteratorCallback(){}
    virtual void callback(const Node& node, int child_position) = 0;
};

#endif // DESTINTREEITERATORCALLBACK_H
