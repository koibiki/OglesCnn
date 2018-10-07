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
 * 前接层维度 (N, 1, 4) , 分类维度不超过4096
 */
public class SoftMax extends Layer {

    private static final String TAG = "SoftMax";

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号

    private int mShaderPro;
    private int mAmount;

    public SoftMax(Context context, String name, Layer preLayer) {
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
        String source = createShaderSource(mOutShape[0]);
        mShaderPro = initCompPro(source);

        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], 1);
    }

    private String createShaderSource(int xSize) {
        String shaderFile = "softmax.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_SOFTMAX_SHADER_HEADER, mAmount, xSize, 1, 1) + source;
    }

    @Override
    public void initialize() {
        initSoftmax();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performComputeWithoutPara(mShaderPro, mPreLayer.getOutTex(), mOutTex, 1, 1, 1);
    }

    @Override
    public float[][][] readResult() {
        float[][][] out = new float[1][1][mAmount];
        DataUtils.readOutput(this, mOut);
        for (int i = 0; i < mAmount; i++) {
            out[0][0][i] = mOut.get(i);
        }
        return out;
    }

}
