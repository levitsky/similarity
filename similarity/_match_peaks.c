#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

typedef struct {
    double (*get_value)(PyArrayObject *, npy_intp);
} Accessor;

static double get_float32(PyArrayObject *array, npy_intp index) {
    return (double)*(float *)PyArray_GETPTR1(array, index);
}

static double get_float64(PyArrayObject *array, npy_intp index) {
    return *(double *)PyArray_GETPTR1(array, index);
}

static int configure_accessor(PyArrayObject **array, Accessor *accessor) {
    int typenum = PyArray_TYPE(*array);
    if (typenum == NPY_FLOAT32) {
        accessor->get_value = get_float32;
        return 0;
    }
    if (typenum == NPY_FLOAT64) {
        accessor->get_value = get_float64;
        return 0;
    }

    PyArrayObject *cast = (PyArrayObject *)PyArray_Cast(*array, NPY_FLOAT64);
    if (cast == NULL) {
        return -1;
    }
    Py_DECREF(*array);
    *array = cast;
    accessor->get_value = get_float64;
    return 0;
}

static npy_intp count_matches(
    PyArrayObject *mz1,
    PyArrayObject *mz2,
    const Accessor *accessor1,
    const Accessor *accessor2,
    double atol,
    double rtol
) {
    npy_intp n1 = PyArray_DIM(mz1, 0);
    npy_intp n2 = PyArray_DIM(mz2, 0);
    npy_intp left = 0;
    npy_intp right = 0;
    npy_intp total = 0;

    for (npy_intp i = 0; i < n1; ++i) {
        double x = accessor1->get_value(mz1, i);
        double lower = nextafter((x - atol) / (1.0 + rtol), -INFINITY);
        double upper = nextafter((x + atol) / (1.0 - rtol), INFINITY);

        while (left < n2 && accessor2->get_value(mz2, left) < lower) {
            ++left;
        }
        if (right < left) {
            right = left;
        }
        while (right < n2 && accessor2->get_value(mz2, right) <= upper) {
            ++right;
        }
        for (npy_intp j = left; j < right; ++j) {
            double y = accessor2->get_value(mz2, j);
            if (fabs(x - y) <= atol + rtol * fabs(y)) {
                ++total;
            }
        }
    }
    return total;
}

static void fill_matches(
    PyArrayObject *mz1,
    PyArrayObject *mz2,
    const Accessor *accessor1,
    const Accessor *accessor2,
    double atol,
    double rtol,
    npy_intp *idx1,
    npy_intp *idx2
) {
    npy_intp n1 = PyArray_DIM(mz1, 0);
    npy_intp n2 = PyArray_DIM(mz2, 0);
    npy_intp left = 0;
    npy_intp right = 0;
    npy_intp out = 0;

    for (npy_intp i = 0; i < n1; ++i) {
        double x = accessor1->get_value(mz1, i);
        double lower = nextafter((x - atol) / (1.0 + rtol), -INFINITY);
        double upper = nextafter((x + atol) / (1.0 - rtol), INFINITY);

        while (left < n2 && accessor2->get_value(mz2, left) < lower) {
            ++left;
        }
        if (right < left) {
            right = left;
        }
        while (right < n2 && accessor2->get_value(mz2, right) <= upper) {
            ++right;
        }
        for (npy_intp j = left; j < right; ++j) {
            double y = accessor2->get_value(mz2, j);
            if (fabs(x - y) <= atol + rtol * fabs(y)) {
                idx1[out] = i;
                idx2[out] = j;
                ++out;
            }
        }
    }
}

static PyObject *match_peaks_sorted(PyObject *self, PyObject *args) {
    PyObject *obj1;
    PyObject *obj2;
    double atol;
    double rtol;

    if (!PyArg_ParseTuple(args, "OOdd", &obj1, &obj2, &atol, &rtol)) {
        return NULL;
    }
    if (rtol >= 1.0) {
        PyErr_SetString(PyExc_ValueError, "rtol must be smaller than 1.0");
        return NULL;
    }

    PyArrayObject *mz1 = (PyArrayObject *)PyArray_FromAny(
        obj1, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *mz2 = (PyArrayObject *)PyArray_FromAny(
        obj2, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    if (mz1 == NULL || mz2 == NULL) {
        Py_XDECREF(mz1);
        Py_XDECREF(mz2);
        return NULL;
    }

    Accessor accessor1;
    Accessor accessor2;
    if (configure_accessor(&mz1, &accessor1) < 0 || configure_accessor(&mz2, &accessor2) < 0) {
        Py_DECREF(mz1);
        Py_DECREF(mz2);
        return NULL;
    }

    npy_intp total = count_matches(mz1, mz2, &accessor1, &accessor2, atol, rtol);
    npy_intp dims[1] = {total};
    PyArrayObject *idx1 = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *idx2 = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (idx1 == NULL || idx2 == NULL) {
        Py_DECREF(mz1);
        Py_DECREF(mz2);
        Py_XDECREF(idx1);
        Py_XDECREF(idx2);
        return NULL;
    }

    fill_matches(
        mz1,
        mz2,
        &accessor1,
        &accessor2,
        atol,
        rtol,
        (npy_intp *)PyArray_DATA(idx1),
        (npy_intp *)PyArray_DATA(idx2)
    );

    Py_DECREF(mz1);
    Py_DECREF(mz2);
    return Py_BuildValue("NN", idx1, idx2);
}

static PyMethodDef MatchPeaksMethods[] = {
    {
        "match_peaks_sorted",
        match_peaks_sorted,
        METH_VARARGS,
        "Return all matching peak index pairs for sorted m/z arrays."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matchpeaksmodule = {
    PyModuleDef_HEAD_INIT,
    "_match_peaks",
    NULL,
    -1,
    MatchPeaksMethods
};

PyMODINIT_FUNC PyInit__match_peaks(void) {
    import_array();
    return PyModule_Create(&matchpeaksmodule);
}