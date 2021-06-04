package edu.bit.practice.common;


import lombok.*;

import java.io.Serializable;


@Data
@Setter
@Getter
@ToString
@AllArgsConstructor
public class BaseResult<T> implements Serializable {

    private Integer code;
    private String message;
    private T data;

}
