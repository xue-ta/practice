package xue.ta.reactor;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
public class ReactorController {

    @GetMapping("/hello/word")
    public Mono<String> hello() {
        return Mono.just("hello word,reactor");
    }
}

