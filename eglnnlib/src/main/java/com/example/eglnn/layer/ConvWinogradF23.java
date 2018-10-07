package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.Numpy;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;

/**
 * 使用 winograd 方式优化的卷积
 */
public class ConvWinogradF23 extends Layer {

    private static final String TAG = "ConvGEMM";

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号

    private final int mKernelAmount;

    private float[][][][] mKennels;
    private int[] mStrides;
    private int[] mKernelShape;
    private int mShaderPro;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private PaddingType mPadType;

    private int mKennelTex;
    private int mPadW, mPadH;

    private ConvWinogradF23(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type) {
        super(context, name, preLayer);
        this.mKernelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKernelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
    }

    public ConvWinogradF23(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type) {
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
        if (outShape[0] / 2 >= 1024) {
            return 1024;
        } else {
            return outShape[0] / 2;
        }
    }

    private int getLocalSizeY(int[] outShape, int xSize) {
        int maxSize = 1024 / xSize;
        if (outShape[1] / 2 <= maxSize) {
            return outShape[1] / 2;
        } else {
            return maxSize;
        }
    }

    private int getLocalSizeZ(int xSize, int ySize) {
        int maxZSize = 1024 / (xSize * ySize) >= 64 ? 64 : 1024 / (xSize * ySize);
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / 2 / xSize);

        int ySize = getLocalSizeY(mOutShape, xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / 2 / ySize);

        int zSize = getLocalSizeZ(xSize, ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) * 2 / 4);

        mKennels = createTestKennels();
//            mKennels = loadKennels();

        mKennelTex = Render.createFloatTextureArray(17, mKernelAmount, Utils.alignBy4(mKernelShape[2]) / 4);
        transferKernelToGgGt(mKennels);

        createShaderParams();
    }

    private void transferKernelToGgGt(float[][][][] mKennels) {
        int kennel_amount = mKennels.length;
        int kennel_channel = mKennels[0].length;

        float[][][][] GgGt = new float[kennel_amount][kennel_channel][4][4];
        float[][] G = new float[][]{{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
        float[][] Gt = Numpy.transpose(G);

        for (int a = 0; a < kennel_amount; a++) {
            for (int c = 0; c < kennel_channel; c++) {
                GgGt[a][c] = Numpy.dot(Numpy.dot(G, mKennels[a][c]), Gt);
            }
        }

        int align_c = Utils.alignBy4(kennel_channel);

        float[] biases = new float[kennel_amount];
        float[][][] GgGt_align = new float[kennel_amount][align_c / 4][16 * 4 + 4]; // depth == 0 处 最后一列第一个为 bias

        for (int a = 0; a < kennel_amount; a++) {
            for (int c = 0; c < kennel_channel; c++) {
                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) {
                        GgGt_align[a][c / 4][(w + h * 4) * 4 + c % 4] = GgGt[a][c][h][w];
                    }
                }
            }
        }
        for (int a = 0; a < kennel_amount; a++) {
            GgGt_align[a][0][16 * 4] = 0.01f * a;
        }

        // 传输到纹理
        for (int a = 0; a < kennel_amount; a++) {
            float[][] kennel = GgGt_align[a];
            int depth = kennel.length;
            for (int c = 0; c < depth; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, 17, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_winograd_2x3.comp", mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[11];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = mType.index;
        mParams[7] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[8] = Utils.alignBy4(mOutShape[2]) / 4;
        mParams[9] = -1 * mPadW;
        mParams[10] = -1 * mPadH;
    }

    private float[][][][] createTestKennels() {
        return TestDataCreator.createConvKennels2(mKernelShape, mKernelAmount);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][][] loadKennels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, mBindLayer);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }

    @Override
    public float[][][] readResult() {
        float[][][] out = new float[mOutShape[2]][mOutShape[1]][mOutShape[0]];
        DataUtils.readOutput(this, mOut);
        DataUtils.transform(out, mOut.array(), mOutShape[0], mOutShape[1], mOutShape[2], mBindLayer);
        return out;
    }

}
