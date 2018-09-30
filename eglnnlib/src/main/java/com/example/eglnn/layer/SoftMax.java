package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_SOFTMAX_SHADER_HEADER;

/**
 * 不超过1024类
 */
public class SoftMax extends Layer {

    private static final String TAG = "SoftMax";

    private int mShaderPro;
    private int mNumGroupsY;
    private int mAmount;

    public SoftMax(Context context, String name, Layer preLayer) {
        super(context, name, preLayer);
        this.mOutShape = calculateSoftMaxShape(preLayer.getOutputShape());
        this.mAmount = preLayer.getOutputShape()[2];
    }

    private int[] calculateSoftMaxShape(int[] preShape) {
        checkShape(preShape[0]);
        checkShape(preShape[1]);
        return new int[]{Utils.alignBy4(preShape[2]) / 4, 1, 1};
    }

    private void checkShape(int shape) {
        if (shape > 1) {
            try {
                throw new Exception("维度不为1");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    private void initSoftmax() {
        String source = createShaderSource(mInShape[2], 1, 1);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], 1);
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String shaderFile = "expand.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_SOFTMAX_SHADER_HEADER, mAmount, xSize, 1, zSize) + source;
    }

    @Override
    public void initialize() {
        initSoftmax();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performComputeWithoutPara(mShaderPro, mPreLayer.getOutTex(), mOutTex, 1, 1, 1);
    }

}
