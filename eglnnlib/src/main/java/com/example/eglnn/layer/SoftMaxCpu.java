package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_SOFTMAX_SHADER_HEADER;

/**
 * 前接层维度 (N, 1, 4) , 分类维度不超过 128 * 4 * 4
 */
public class SoftMaxCpu extends Layer {

    private static final String TAG = "SoftMax";

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号
    private float[] mOutArray;

    private int mShaderPro;
    private int mAmount;

    public SoftMaxCpu(Context context, String name, Layer preLayer) {
        super(context, name, preLayer);
        if (preLayer instanceof Flat) {
            this.mAmount = ((Flat) preLayer).getAmount();
        } else {
            this.mAmount = mInShape[0] * mInShape[1] * mInShape[2];
        }
        checkShape(mInShape[1], 1);
        checkShape(mInShape[2], 4);
        this.mOutShape = mInShape;
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
    }

    private void checkShape(int shape, int i) {
        if (shape != i) {
            try {
                throw new Exception("维度不为" + i);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void initSoftmax() {

    }


    @Override
    public void initialize() {
        initSoftmax();
    }

    @Override
    protected void bindTextureAndBuffer() {

    }

    @Override
    protected void actualForwardProc(float[][] input) {
        mOutArray = new float[mAmount];
        DataUtils.readOutput(mPreLayer, mOut, mOutShape[0], mOutShape[1]);
        float sum = 0f;
        for (int i = 0; i < mAmount; i++) {
            float v = (float) Math.exp(mOut.get(i));
            mOutArray[i] = v;
            sum += v;
        }
        for (int i=0; i < mAmount; i++) {
            mOutArray[i] = mOutArray[i]  / sum;
        }
    }

    @Override
    public float[] readResult() {
        return mOutArray;
    }

}
