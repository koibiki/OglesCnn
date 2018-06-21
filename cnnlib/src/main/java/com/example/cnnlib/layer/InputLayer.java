package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.SortUtils;

import java.nio.FloatBuffer;

import static com.example.cnnlib.utils.DataUtils.createInputBuffer;

public class InputLayer extends Layer {

    private float[][] mInput;

    public InputLayer(Context context, int[] shape) {
        super(context, shape);
    }

    @Override
    public void initialize() {
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture();
        initInput();
    }

    private void initInput() {
        mInput = createInputBuffer(mOutputShape);
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        initInput();
        int channel = mOutputShape[2];
        int[] count = SortUtils.getCount(channel);
        for (int i = 0; i < count[0]; i++) {
            writeInput(mInput, mOutTex, i);
        }
    }

    private void writeInput(float[][] input, int texID, int index) {
        int width = mOutputShape[0];
        int[] indexes = SortUtils.getXYIndex(width, index);
        int height = mOutputShape[1];
        int channel = mOutputShape[2];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        float[] localInput = new float[width * height * 4];
        for (int i = 0; i < width * height; i++) {
            for (int ii = 0; ii < 4; ii++) {
                if (index * 4 + ii < channel) {
                    localInput[i * 4 + ii] = input[index * 4 + ii][i];
                } else {
                    localInput[i * 4 + ii] = 0;

                }
            }
        }
        ComputeRender.transferToTexture(FloatBuffer.wrap(localInput), texID, startX, startY, width, height);
    }

}
