package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;

/**
 * 按照卷积逻辑进行卷积
 */
public class Conv extends Layer {

    private static final String TAG = "ConvGEMM";

    private final int mKernelAmount;
    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号

    private float[][][] mKernels;
    private int[] mStrides;
    private int[] mKernelShape;
    private int mShaderPro;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private PaddingType mPadType;

    private int mKernelTex;
    private int mPadW, mPadH;

    private Conv(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type) {
        super(context, name, preLayer);
        this.mKernelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKernelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
    }

    public Conv(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type) {
        this(context, name, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kAmount);
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
//        this.mBindLayer = Utils.alignBy4(mOutShape[2]) / 4;
    }

    private int[] calculateConvShapeByType(int kennelAmount) {
        int width, height;
        if (mPadType == PaddingType.SAME) {
            width = (int) Math.ceil(mInShape[0] * 1.0f / mStrides[0]);
            mPadW = ((width - 1) * mStrides[0] + mKernelShape[0] - mInShape[0]) / 2;
            height = (int) Math.ceil(mInShape[1] * 1.0f / mStrides[1]);
            mPadH = ((height - 1) * mStrides[1] + mKernelShape[1] - mInShape[1]) / 2;
        } else {
            width = (int) Math.ceil((mInShape[0] - mKernelShape[0] + 1) * 1.0f / mStrides[0]);
            height = (int) Math.ceil((mInShape[1] - mKernelShape[1] + 1) * 1.0f / mStrides[1]);
        }
        return new int[]{width, height, kennelAmount};
    }

    private int getLocalSizeX(int[] outShape) {
        if (outShape[0] >= GLES31BackEnv.getMaxWorkGroupSize()) {
            return GLES31BackEnv.getMaxWorkGroupSize();
        } else {
            return outShape[0];
        }
    }

    private int getLocalSizeY(int[] outShape, int xSize) {
        int maxSize = GLES31BackEnv.getMaxWorkGroupSize() / xSize;
        if (outShape[1] <= maxSize) {
            return outShape[1];
        } else {
            return maxSize;
        }
    }

    private int getLocalSizeZ(int xSize, int ySize) {
        int maxZSize = GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize) >= 64 ? 64 : GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize);
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / xSize);

        int ySize = getLocalSizeY(mOutShape, xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / ySize);

        int zSize = getLocalSizeZ(xSize, ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        mKernels = createTestKernels();
//            mKernels = loadKennels();

        mKernelTex = Render.createFloatTextureArray(mKernelShape[0] * mKernelShape[1] + 1, mKernelAmount, Utils.alignBy4(mKernelShape[2]) / 4);
        transferToKennelTex();
        createShaderParams();
    }

    private void transferToKennelTex() {
        for (int a = 0; a < mKernelAmount; a++) {
            float[][] kennel = mKernels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKernelTex, 0, a, c, mKernelShape[0] * mKernelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_tex.comp", mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[15];
        mParams[0] = mKernelShape[0];
        mParams[1] = mKernelShape[1];
        mParams[2] = mKernelShape[2];
        mParams[3] = mInShape[0];
        mParams[4] = mInShape[1];
        mParams[5] = mInShape[2];
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mType.index;
        mParams[12] = Utils.alignBy4(mInShape[2]);
        mParams[13] = -1 * mPadW;
        mParams[14] = -1 * mPadH;
    }

    private float[][][] createTestKernels() {
        return TestDataCreator.createConvKennels(mKernelShape, mKernelAmount);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][] loadKernels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKernelTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }

    @Override
    public float[] readResult() {
        float[][][] out = new float[mOutShape[2]][mOutShape[1]][mOutShape[0]];
        DataUtils.readOutput(this, mOut, mOutShape[0], mOutShape[1]);
        DataUtils.transform(out, mOut.array(), mOutShape[0], mOutShape[1], mOutShape[2], mBindLayer);
        return mOut.array();
    }
}
