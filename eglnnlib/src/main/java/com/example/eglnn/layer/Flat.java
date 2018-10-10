package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;


/**
 * 前接层 shape 为 （1，1，N）
 */
public class Flat extends Layer {

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mShaderPro;
    private int mBindLayer;         // 纹理绑定深度编号

    private int mAmount;
    private int mNumGroupX;

    public Flat(Context context, String name, Layer preLayer) {
        super(context, name, preLayer);
        if (preLayer instanceof FullConnSSBO) {
            this.mAmount = ((FullConnSSBO) preLayer).getKernelAmount();
        } else {
            this.mAmount = mInShape[2];
        }
        this.mOutShape = new int[]{Utils.alignBy4(mAmount) / 4, 1, 4};
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
    }

    public int getAmount() {
        return mAmount;
    }

    @Override
    public void initialize() {
        int xSize;
        if (mOutShape[0] <= GLES31BackEnv.getMaxWorkGroupSize()) {
            xSize = mOutShape[0];
            mNumGroupX = 1;
        } else {
            xSize = GLES31BackEnv.getMaxWorkGroupSize();
            mNumGroupX = (int) Math.ceil(mOutShape[0] * 1.0f / GLES31BackEnv.getMaxWorkGroupSize());
        }

        String source = createShaderSource(xSize);
        mShaderPro = initCompPro(source);

        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(Utils.alignBy4(mOutShape[0]), 1, Utils.alignBy4(mOutShape[2]) / 4);
    }

    private String createShaderSource(int xSize) {
        String shaderFile = "flat.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, 1, 1) + source;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, mBindLayer);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performComputeWithoutPara(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupX, 1, 1);
    }

    @Override
    public float[] readResult() {
        DataUtils.readOutput(this, mOut, mOutShape[0], mOutShape[1]);
        return  mOut.array();
    }

}
