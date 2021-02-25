package edu.bit.practice.handler;


public interface HandlerChain {
    public HandlerChain addLast(Handler handler);
}
