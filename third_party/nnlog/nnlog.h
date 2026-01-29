#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace nnlog
{

    enum class Level
    {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
        Fatal = 5
    };

    struct Config
    {
        Level minLevel = Level::Info;
        bool showTimestamp = true;
        bool showThreadId = false;
        bool useStderrForError = true;
    };

    inline Config &config()
    {
        static Config cfg;
        return cfg;
    }

    inline const char *toString(Level level)
    {
        switch (level)
        {
        case Level::Trace:
            return "TRACE";
        case Level::Debug:
            return "DEBUG";
        case Level::Info:
            return "INFO";
        case Level::Warn:
            return "WARN";
        case Level::Error:
            return "ERROR";
        case Level::Fatal:
            return "FATAL";
        default:
            return "INFO";
        }
    }

    inline std::mutex &outputMutex()
    {
        static std::mutex m;
        return m;
    }

    inline std::string formatTimestamp()
    {
        using clock = std::chrono::system_clock;
        const auto now = clock::now();
        const auto t = clock::to_time_t(now);

        std::tm tm;
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm, "%H:%M:%S");
        return oss.str();
    }

    class LogLine
    {
    public:
        LogLine(Level level, const char *tag, const char *func, const char *file, int line)
            : level_(level), tag_(tag ? tag : ""), func_(func ? func : ""), file_(file ? file : ""), line_(line)
        {
        }

        ~LogLine()
        {
            if (static_cast<int>(level_) < static_cast<int>(config().minLevel))
            {
                return;
            }

            std::string msg = ss_.str();
            if (!msg.empty() && msg.back() != '\n')
            {
                msg.push_back('\n');
            }

            std::lock_guard<std::mutex> lock(outputMutex());

            std::ostream *out = &std::cout;
            if (config().useStderrForError && (level_ == Level::Error || level_ == Level::Fatal))
            {
                out = &std::cerr;
            }

            if (config().showTimestamp)
            {
                (*out) << formatTimestamp() << " ";
            }
            (*out) << "[" << toString(level_) << "]";
            if (!tag_.empty())
            {
                (*out) << "[" << tag_ << "]";
            }
            if (!func_.empty())
            {
                (*out) << func_ << ": ";
            }
            if (config().showThreadId)
            {
                (*out) << "(t=" << std::this_thread::get_id() << ") ";
            }

            (*out) << msg;
            out->flush();

            if (level_ == Level::Fatal)
            {
                std::terminate();
            }
        }

        LogLine(const LogLine &) = delete;
        LogLine &operator=(const LogLine &) = delete;

        template <typename T>
        LogLine &operator<<(const T &value)
        {
            ss_ << value;
            return *this;
        }

        using Manip = std::ostream &(*)(std::ostream &);
        LogLine &operator<<(Manip manip)
        {
            manip(ss_);
            return *this;
        }

    private:
        Level level_;
        std::string tag_;
        std::string func_;
        [[maybe_unused]] std::string file_;
        [[maybe_unused]] int line_;
        std::ostringstream ss_;
    };

} // namespace nnlog

// Convenience macros. These create a temporary LogLine so `NNLOG_INFO() << ...` works.
#define NNLOG_TRACE(TAG) ::nnlog::LogLine(::nnlog::Level::Trace, (TAG), __FUNCTION__, __FILE__, __LINE__)
#define NNLOG_DEBUG(TAG) ::nnlog::LogLine(::nnlog::Level::Debug, (TAG), __FUNCTION__, __FILE__, __LINE__)
#define NNLOG_INFO(TAG) ::nnlog::LogLine(::nnlog::Level::Info, (TAG), __FUNCTION__, __FILE__, __LINE__)
#define NNLOG_WARN(TAG) ::nnlog::LogLine(::nnlog::Level::Warn, (TAG), __FUNCTION__, __FILE__, __LINE__)
#define NNLOG_ERROR(TAG) ::nnlog::LogLine(::nnlog::Level::Error, (TAG), __FUNCTION__, __FILE__, __LINE__)
#define NNLOG_FATAL(TAG) ::nnlog::LogLine(::nnlog::Level::Fatal, (TAG), __FUNCTION__, __FILE__, __LINE__)
