package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;


/**
 * 前接层 shape 为 （1，1，N）
 */
public class Flat extends Layer {

    private int mAmount;
    private int mNumGroupX;
    private int mShaderPro;

    public Flat(Context context, String name, Layer preLayer) {
        super(context, name, preLayer);
        if (preLayer instanceof FullConnSSBO) {
            this.mAmount = ((FullConnSSBO) preLayer).getKernelAmount();
        } else {
            this.mAmount = mInShape[2];
        }
        this.mOutShape = new int[]{Utils.alignBy4(mAmount) / 4, 1, 4};
    }

    public int getAmount() {
        return mAmount;
    }

    @Override
    public void initialize() {
        int xSize;
        if (mOutShape[0] <= 1024) {
            xSize = mOutShape[0];
            mNumGroupX = 1;
        } else {
            xSize = 1024;
            mNumGroupX = (int) Math.ceil(mOutShape[0] * 1.0f / 1024);
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
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performComputeWithoutPara(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupX, 1, 1);
    }

}
