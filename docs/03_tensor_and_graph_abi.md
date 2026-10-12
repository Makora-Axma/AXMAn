# Спецификация C-ABI: Тензоры и ориентированные гиперграфы (AXMANet)

Для того чтобы архитектура **AXMANet** могла бесшовно вызываться из C++, Java (Panama), Python и Rust, определен стабильный бинарный C-интерфейс (`include/axman_c_api.h`).

---

## 1. Актуальный заголовочный контракт C-ABI (`include/axman_c_api.h`)

```c
#ifndef AXMAN_C_API_H
#define AXMAN_C_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Dynamic Library Export / Import Symbol Macro                              */
/* -------------------------------------------------------------------------- */

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(AXMAN_EXPORTS) || defined(AXMAN_BUILDING_DLL)
    #define AXMAN_API __declspec(dllexport)
  #elif defined(AXMAN_STATIC) || defined(AXMAN_STATIC_LIB)
    #define AXMAN_API
  #else
    #define AXMAN_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define AXMAN_API __attribute__((visibility("default")))
  #else
    #define AXMAN_API
  #endif
#endif

/* -------------------------------------------------------------------------- */
/* Version & Dimension Constants                                              */
/* -------------------------------------------------------------------------- */

#define AXMAN_VERSION_MAJOR 1
#define AXMAN_VERSION_MINOR 0
#define AXMAN_VERSION_PATCH 0
#define AXMAN_VERSION_STRING "1.0.0"

/**
 * @brief Maximum supported hyperedge arity (k_max = 16) in a static SRAM kernel tile.
 * Hyperedges with higher arity must undergo canonical currying decomposition.
 */
#define AXMAN_MAX_HYPEREDGE_ARITY 16

/* -------------------------------------------------------------------------- */
/* Status and Error Codes                                                     */
/* -------------------------------------------------------------------------- */

typedef enum {
    AXMAN_SUCCESS                 = 0,
    AXMAN_ERR_INVALID_ARGUMENT    = 1,
    AXMAN_ERR_OUT_OF_MEMORY       = 2,
    AXMAN_ERR_DEVICE_UNAVAILABLE  = 3,
    AXMAN_ERR_INFERENCE_FAILURE   = 4,
    AXMAN_ERR_NULL_POINTER        = 5,
    AXMAN_ERR_ARITY_OVERFLOW      = 6,
    AXMAN_ERR_INDEX_OUT_OF_BOUNDS = 7,
    AXMAN_ERR_INVALID_TOPOLOGY    = 8,
    AXMAN_ERR_INTERNAL            = 99
} axman_status_t;

/* -------------------------------------------------------------------------- */
/* Tensor Metadata & Data Types                                               */
/* -------------------------------------------------------------------------- */

typedef enum {
    AXMAN_DTYPE_FLOAT32  = 0,
    AXMAN_DTYPE_BFLOAT16 = 1,
    AXMAN_DTYPE_INT64    = 2,
    AXMAN_DTYPE_INT32    = 3,
    AXMAN_DTYPE_FLOAT16  = 4
} axman_dtype_t;

typedef enum {
    AXMAN_DEVICE_CPU  = 0,
    AXMAN_DEVICE_CUDA = 1
} axman_device_t;

/**
 * @brief Zero-copy descriptor for dense contiguous tensors.
 */
typedef struct {
    void* data;              /**< Pointer to contiguous memory buffer */
    int64_t* shape;          /**< Dimension array [ndim] */
    size_t ndim;             /**< Number of dimensions (rank) */
    int32_t dtype;           /**< Data type (axman_dtype_t) */
    int32_t device;          /**< Device type (axman_device_t) */
} axman_tensor_t;

/* -------------------------------------------------------------------------- */
/* Port-Hypergraph Topology (CSR / COO Layout)                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Directed Port-Hypergraph G = (V, E, P) in Compressed Sparse Row (CSR) format.
 *
 * Memory ownership contract:
 * All arrays (edge_sources, edge_ports, edge_targets, edge_offsets) and node_features
 * are borrowed views allocated and managed by the caller. AXMAN functions do not free
 * or reallocate these buffers.
 */
typedef struct {
    size_t num_nodes;                /**< Number of typed objects / AST nodes in V */
    size_t num_hyperedges;           /**< Number of morphisms / hyperedges in E */

    axman_tensor_t node_features;    /**< Node feature matrix: [num_nodes, node_feat_dim] */

    const int64_t* edge_sources;     /**< Flat contiguous array of source premise node indices */
    const int32_t* edge_ports;       /**< Positional port numbers (de Bruijn binding indices 0..k-1) */
    const int64_t* edge_targets;     /**< Target node index for each hyperedge [num_hyperedges] */
    const size_t* edge_offsets;      /**< CSR offsets into edge_sources [num_hyperedges + 1] */
} axman_hypergraph_t;

/* -------------------------------------------------------------------------- */
/* Model Output Structures                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Output container returned by model forward pass.
 *
 * Memory ownership contract:
 * Buffers are allocated by AXMAN runtime and must be released by calling
 * axman_output_free(&output).
 */
typedef struct {
    axman_tensor_t policy_logits;    /**< Morphism transition & premise selection logits */
    float value_estimate;            /**< Distance-to-closure value scalar in [0.0, 1.0] */
    axman_tensor_t latent_embedding; /**< Latent state invariant representation */
} axman_output_t;

/**
 * @brief Opaque handle to the instantiated AXMANet model instance.
 */
typedef struct axman_model_opaque* axman_model_handle_t;

/* -------------------------------------------------------------------------- */
/* Model Lifecycle Functions                                                  */
/* -------------------------------------------------------------------------- */

AXMAN_API axman_status_t axman_model_create(
    const char* model_path,
    const char* config_json,
    axman_model_handle_t* out_handle
);

AXMAN_API axman_status_t axman_model_forward(
    axman_model_handle_t handle,
    const axman_hypergraph_t* input_graph,
    axman_output_t* out_result
);

AXMAN_API axman_status_t axman_model_destroy(
    axman_model_handle_t handle
);

AXMAN_API void axman_output_free(
    axman_output_t* output
);

/* -------------------------------------------------------------------------- */
/* Graph Validation & Structural Hashing                                      */
/* -------------------------------------------------------------------------- */

AXMAN_API axman_status_t axman_hypergraph_validate(
    const axman_hypergraph_t* graph
);

AXMAN_API axman_status_t axman_hypergraph_hash(
    const axman_hypergraph_t* graph,
    uint64_t* out_hash
);

/* -------------------------------------------------------------------------- */
/* Diagnostics & Metadata                                                     */
/* -------------------------------------------------------------------------- */

AXMAN_API const char* axman_status_to_string(axman_status_t status);
AXMAN_API const char* axman_get_last_error_message(void);
AXMAN_API const char* axman_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AXMAN_C_API_H */
```

---

## 2. Спецификация Port-Hypergraph в представлении CSR / COO

Ориентированный мультигиперграф $G = (V, E, P)$ формализует доказательное состояние:
* **$V$ (Вершины/Объекты):** Типы, термы, пропозиции и свободные гипотезы контекста. Количество вершин задается полем `num_nodes`.
* **$E$ (Гиперребра/Морфизмы):** Правила вывода, тактические применения и конструкторы типов. Каждое гиперребро $e \in E$ связывает кортеж предпосылок $(u_1, \dots, u_k)$ с одной целевой вершиной $v \in V$.
* **$P$ (Порты связывания):** Позиционные номера аргументов в сигнатуре и глубины связывания де Брёйна ($0 \le p_j < k$).

### 2.1. Топологические массивы CSR

Для максимальной производительности GPU и исключения фрагментации памяти граф упакован в 4 непрерывных одномерных массива:

1. **`edge_offsets`** (размер `num_hyperedges + 1`):
   Массив смещений CSR. Для гиперребра с индексом $i \in [0, \text{num\_hyperedges}-1]$ диапазон его источников в `edge_sources` и `edge_ports` определяется полуинтервалом:
   $$[\text{edge\_offsets}[i], \; \text{edge\_offsets}[i+1])$$
   Арность гиперребра $k_i$ вычисляется тривиально за $O(1)$:
   $$k_i = \text{edge\_offsets}[i+1] - \text{edge\_offsets}[i]$$
   Инвариант: `edge_offsets[0] == 0` и монотонное неубывание `edge_offsets[i] <= edge_offsets[i+1]`.

2. **`edge_sources`** (размер `edge_offsets[num_hyperedges]`):
   Плоский непрерывный массив индексов вершин-посылок ($u_j \in [0, \text{num\_nodes}-1]$).

3. **`edge_ports`** (размер `edge_offsets[num_hyperedges]`):
   Массив позиционных портов де Брёйна, строго сопоставленный элементам `edge_sources`.

4. **`edge_targets`** (размер `num_hyperedges`):
   Индекс целевой вершины ($v_i \in [0, \text{num\_nodes}-1]$) для каждого гиперребра $i$.

---

## 3. Инвариант лимита арности (Arity Cap) и Декомпозиция Каррирования

### 3.1. Проблема Hyperedge Arity Explosion
При компиляции математических выражений из Lean 4 функции и теоремы могут иметь десятки аргументов (явные параметры, гипотезы, неявные инстансы тайпклассов вроде `[Decidable p]`, `[CommRing R]`).
Если в ядро Flash-Attention передать гиперребро с $k = 40$, статический тайл в разделяемой памяти GPU (SRAM) испытает взрывное переполнение, либо произойдет сильная дивергенция потоков (Warp Divergence).

### 3.2. Архитектурный лимит $k_{max} = 16$
Константа `AXMAN_MAX_HYPEREDGE_ARITY = 16` жестко фиксирует максимальное количество источников на одно гиперребро в памяти ядра:
$$\forall i \in [0, \text{num\_hyperedges}-1]: \quad k_i \le 16$$

### 3.3. Канонический алгоритм декомпозиции каррирования (Currying Pass)
Если гиперребро $e = (u_0, u_1, \dots, u_{k-1}) \xrightarrow{p_0, \dots, p_{k-1}} v$ имеет арность $k > 16$:
Оно преобразуется препроцессором в цепочку связанных гиперребер с созданием вспомогательных промежуточных узлов (Auxiliary Intermediate Nodes):
1. Первые 15 аргументов $(u_0, \dots, u_{14})$ образуют промежуточное гиперребро $e^{(1)} \to \alpha_1$, где $\alpha_1$ — синтезированный узел частичного применения.
2. Промежуточный узел $\alpha_1$ становится нулевым источником следующего гиперребра вместе со следующими аргументами: $(\alpha_1, u_{15}, \dots, u_{29}) \to \alpha_2$.
3. Процесс повторяется до тех пор, пока финальный сегмент не соединится с целевым узлом $v$.

Каждое ребро в декомпозированной цепочке строго удовлетворяет условию $k \le 16$, сохраняя точную семантику каррирования:
$$f(x_1, \dots, x_k) \equiv (\dots (f(x_1, \dots, x_{15}))(x_{16}, \dots) \dots)$$

---

## 4. Модель управления памятью, владения и Zero-Copy контракт (Memory Ownership Model)

Для обеспечения времени маршалинга графовых данных менее 1 микросекунды и предотвращения утечек памяти через FFI в AXMANet принята строгая асимметричная модель владения:

### 4.1. Входные структуры (Input Borrowed Views)
* **Принцип Caller-Allocated / Borrowed View:**  
  Все массивы входного гиперграфа `axman_hypergraph_t` (`edge_sources`, `edge_ports`, `edge_targets`, `edge_offsets`) и тензор признаков узлов `node_features.data` **принадлежат вызывающей стороне**.
* **Гарантия неизменяемости (Immutability Contract):**  
  Функции инференса (`axman_model_forward`), валидаторы и хэшеры принимают входные структуры как `const axman_hypergraph_t*`. Библиотека AXMAN **никогда не освобождает** (`free`), **не реаллоцирует** и **не мутирует** входные буферы.
* **Требование к времени жизни (Lifetime Requirement):**  
  Буферы вызывающей стороны должны оставаться валидными и неизменными только на время синхронного выполнения вызова C-ABI (`forward` или `validate`). По завершении функции вызывающая сторона вольна переиспользовать или освободить буферы.
* **Многопоточность (Thread-Safety & Reentrancy):**  
  Поскольку входные структуры являются строго неизменяемыми заимствованными представлениями, несколько потоков могут параллельно вызывать инференс модели над одним и тем же экземпляром `axman_hypergraph_t` без копирования данных.

### 4.2. Выходные структуры (Runtime-Allocated Outputs)
* **Принцип Runtime-Allocated Heap Ownership:**  
  Выходная структура `axman_output_t` (`policy_logits.data`, `policy_logits.shape`, `latent_embedding.data`, `latent_embedding.shape`) аллоцируется вычислительным ядром AXMANet в куче (Heap).
* **Передача владения (Ownership Handover):**  
  При успешном возврате `AXMAN_SUCCESS` из `axman_model_forward` вызывающая сторона получает полное и исключительное право владения выходными тензорами.
* **Обязательное освобождение через C-ABI:**  
  Вызывающая сторона обязана освободить выходную память вызовом `axman_output_free(&output)`. Прямой вызов `free()` из внешних языков или сред с другими рантаймами C (например, MSVCRT vs UCRT) запрещен во избежание повреждения кучи (Heap Corruption).
* **Идемпотентность и защита от Double-Free:**  
  Функция `axman_output_free` обнуляет освобожденные указатели (`data = NULL`, `shape = NULL`, `ndim = 0`). Повторный вызов `axman_output_free` над уже освобожденной структурой безопасен и не приводит к аварийному завершению.

### 4.3. Сопоставление с языковыми обёртками

| Язык | Представление входа (Zero-Copy) | Управление выходом | Механизм освобождения |
| :--- | :--- | :--- | :--- |
| **Rust** | `&HypergraphState` (неизменяемая ссылка) | RAII `struct ModelOutput` | Реализация трейта `Drop` с вызовом `axman_output_free` |
| **C++** | `std::span<const int64_t>`, `const Hypergraph&` | RAII класс `axman::InferenceResult` | Деструктор вызывает `axman_output_free` |
| **Python** | `torch.Tensor` / NumPy массив через FFI buffer protocol | PyO3 класс `ModelOutput` | Метод `__del__` / PyCapsule деструктор |
| **Java** | `MemorySegment` в `Arena.ofConfined()` | Защищенный `MemorySegment` | Закрытие `Arena` (try-with-resources) |

---

## 5. Детерминированная валидация и 64-битное структурное хэширование графа

Для защиты от сбоев сегментации (SegFault) на границе C-ABI и обеспечения сверхбыстрого дедуплицирования графов в поисковых движках MCTS и кэшах инференса реализованы функции `axman_hypergraph_validate` и `axman_hypergraph_hash`.

### 5.1. Полная валидация топологических инвариантов
Функция `axman_hypergraph_validate(const axman_hypergraph_t* graph)` выполняет строгую проверку $O(|V| + |E| + |P|)$:
1. **Проверка указателей:** Проверка `graph != NULL`, а при `num_hyperedges > 0` — валидность `edge_offsets`, `edge_targets`, `edge_sources`, `edge_ports`.
2. **Корректность структуры CSR:** Базовое смещение `edge_offsets[0] == 0`, монотонность `edge_offsets[i] <= edge_offsets[i+1]`.
3. **Лимит арности (Arity Cap):** Каждое гиперребро строго удовлетворяет $\text{edge\_offsets}[i+1] - \text{edge\_offsets}[i] \le 16$.
4. **Границы индексов:** Все вершины-посылки и целевые вершины находятся в диапазоне $[0, \text{num\_nodes}-1]$. При выходе за границу возвращается `AXMAN_ERR_INDEX_OUT_OF_BOUNDS`.
5. **Валидность портов:** Номера портов де Брёйна неотрицательны.
6. **Тензор признаков:** При передаче `node_features.data` проверяются `ndim == 2`, `shape[0] == num_nodes`, `shape[1] > 0` и допустимый тип `dtype`.

### 5.2. Алгоритм де Брёйн структурного хэширования (XXH64)
Функция `axman_hypergraph_hash(const axman_hypergraph_t* graph, uint64_t* out_hash)`:
1. Выполняет предварительную валидацию графа. При обнаружении ошибок вычисление хэша прерывается с возвратом соответствующего кода ошибки.
2. Пропускает топологию через потоковый алгоритм **XXH64** с фиксированным начальным сидом `0x41584D414EULL` (`"AXMAN"`):
   - Заголовок: кортеж `(num_nodes, num_hyperedges, total_sources)`.
   - Непрерывный срез `edge_offsets` (длина `num_hyperedges + 1`).
   - Массив `edge_targets` (длина `num_hyperedges`).
   - Массив `edge_sources` (длина `total_sources`).
   - Массив `edge_ports` (длина `total_sources`, кодирующий позиционную привязку де Брёйна).
   - Сигнатура признаков узлов (форма, ранг, тип данных).
3. **Свойства хэша:**
   - **Строгий детерминизм:** Графы с идентичной топологией гарантированно производят побитово идентичный 64-битный хэш.
   - **Лавинный эффект (Avalanche Effect):** Изменение даже одного индекса источника, цели или порта де Брёйна изменяет в среднем более 50% бит результирующего хэша.
   - **Интеграция с LRU-кэшем:** Позволяет реализовать в `libaxman` субмикросекундный поиск в кэше состояний без повторного запуска нейросетевых слоев.
