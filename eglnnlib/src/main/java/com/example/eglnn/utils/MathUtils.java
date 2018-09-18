package com.example.eglnn.utils;

public class MathUtils {

    public static int getPowerBy2(int num) {
        int count = 0;
        if (num >= 2) {
            num = num / 2;
            count++;
            count += getPowerBy2(num);
        }
        return count;
    }

}
