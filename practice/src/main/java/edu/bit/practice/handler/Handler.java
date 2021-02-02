package edu.bit.practice.handler;

public interface Handler<T> {
    void handleRequest(T t);
}
