package edu.bit.practice;

import edu.bit.practice.netty.EchoServer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;



@EnableScheduling
@SpringBootApplication
public class PracticeApplication {


	public static void main(String[] args) {
		SpringApplication.run(PracticeApplication.class, args);

	}

}
