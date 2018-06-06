package com.example.cnnlib.model;

import android.graphics.Point;

import com.example.cnnlib.utils.Constants;

public class LayerParams {

    public final int[] inputShape = new int[3];     // w,h,c
    public final int[] outputShape = new int[3];    // w,h,c


    public LayerParams(int inputWidth, int inputHeight, int inputChannel, int outputWidth, int outputHeight, int outputChannel) {
        this.inputShape[0] = inputWidth;
        this.inputShape[1] = inputHeight;
        this.inputShape[2] = inputChannel;
        this.outputShape[0] = outputWidth;
        this.outputShape[1] = outputHeight;
        this.outputShape[2] = outputChannel;
    }

    public int getInputSize() {
        return inputShape[0] * inputShape[1] * inputShape[2];
    }

    public int getOutputSize() {
        return outputShape[0] * outputShape[1] * outputShape[2];
    }

    private int getXCount(int width, int channel) {
        int maxXCount = Constants.S_TEXTURE_SIZE / width;
        int realCount = channel / 4;
        return realCount <= maxXCount ? realCount : maxXCount;
    }

    private int getYCount(int width, int height, int channel) {
        int maxXCount = Constants.S_TEXTURE_SIZE / width;
        int realCount = (int) Math.ceil(channel * 1.0d / 4);
        return (int) Math.ceil(realCount * 1.0d / maxXCount);
    }

    public Point getStartIndex(int index, int width, int height, int channel) {
        Point point = new Point();
        point.x = index % getXCount(width, channel) * width;
        point.y = index / getXCount(width, channel) * height;
        return point;
    }

    public Shape getInputShape() {
        return getShape(inputShape[2]);
    }

    public Shape getOutputShape() {
        return getShape(outputShape[2]);
    }

    private Shape getShape(int channel) {
        Shape shape = new Shape();
        shape.setCount(channel / 4);
        shape.setRemain(channel % 4);
        return shape;
    }

    public class Shape {
        private int count;
        private int remain;

        public int getCount() {
            return count;
        }

        public void setCount(int count) {
            this.count = count;
        }

        public int getRemain() {
            return remain;
        }

        public void setRemain(int remain) {
            this.remain = remain;
        }
    }

}